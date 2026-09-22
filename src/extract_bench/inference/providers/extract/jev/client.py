"""OpenRouter Decisions client with persistent, concurrent-safe spend accounting."""

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import requests


class JevClient:
    def __init__(self, directory, label="", budget=9.0, model="typesafe/jev-1.13"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.label, self.budget, self.model = label, budget, model
        self.calls = 0
        self.cost = 0.0
        self.max_calls = 2000
        self.deadline = time.monotonic() + 1800
        self.session = requests.Session()
        self.session.headers["Authorization"] = "Bearer " + os.environ["OPENROUTER_API_KEY"]
        with self.db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, label TEXT, "
                "status TEXT, cost REAL, tokens INTEGER, request_hash TEXT)"
            )

    def db(self):
        return sqlite3.connect(self.directory / "budget.sqlite", timeout=60)

    def summary(self):
        with self.db() as db:
            return dict(
                zip(
                    ("attempts", "accounted_usd", "input_tokens"),
                    db.execute("SELECT COUNT(*), COALESCE(SUM(cost),0), COALESCE(SUM(tokens),0) FROM calls").fetchone(),
                    strict=True,
                )
            )

    def decide(self, state, questions):
        if not questions:
            return {}
        # Small batches keep alpha endpoint requests below the 32K context limit.
        # State is never silently truncated: variants must retrieve/chunk it.
        batches, current = [], {}
        for key, question in questions.items():
            trial = {**current, key: question}
            size = len(json.dumps({"state": state, "questions": trial}, ensure_ascii=False).encode())
            if current and (size > 26000 or len(trial) > 24):
                batches.append(current)
                current = {key: question}
            else:
                current = trial
        if current:
            batches.append(current)
        answers = {}
        for batch in batches:
            for attempt in range(3):
                try:
                    answers.update(self._request(state, batch))
                    break
                except requests.RequestException as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    transient = isinstance(exc, (requests.Timeout, requests.ConnectionError)) or status in (
                        429,
                        500,
                        502,
                        503,
                        504,
                    )
                    if not transient or attempt == 2:
                        raise
                    # Retry only the failed batch; earlier successful batches are retained.
                    # Each attempt reserves its own cost in the shared ledger.
                    time.sleep(2**attempt)
        return answers

    def _request(self, state, questions):
        if self.calls >= self.max_calls or time.monotonic() >= self.deadline:
            raise RuntimeError("Jev per-document request/time limit reached")
        body = {"model": self.model, "state": state, "questions": questions}
        encoded = json.dumps(body, ensure_ascii=False).encode()
        if len(encoded) > 100000:
            raise ValueError("Jev request too large; retrieve or chunk state before sending")
        # UTF-8 byte bound plus protocol allowance; unresolved attempts retain reservation.
        reserve = (len(encoded) + 4096) * 0.000000042
        digest = hashlib.sha256(encoded).hexdigest()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            spent = db.execute("SELECT COALESCE(SUM(cost),0) FROM calls").fetchone()[0]
            if spent + reserve > self.budget:
                raise RuntimeError(f"Shared Jev budget exhausted: ${spent:.4f}/${self.budget:.2f}")
            call_id = db.execute(
                "INSERT INTO calls(label,status,cost,tokens,request_hash) VALUES (?,?,?,?,?)",
                (self.label, "reserved", reserve, 0, digest),
            ).lastrowid
        self.calls += 1
        self.cost += reserve
        started = time.monotonic()
        archive = {"request": body, "label": self.label}
        try:
            response = self.session.post("https://openrouter.ai/api/alpha/decisions", json=body, timeout=90)
            archive.update(status_code=response.status_code, elapsed=time.monotonic() - started)
            try:
                archive["response"] = response.json()
            except ValueError:
                archive["response_text"] = response.text
                raise
            (self.directory / f"call_{call_id:07d}.json").write_text(json.dumps(archive, ensure_ascii=False))
            response.raise_for_status()
            result = archive["response"]
            usage = result.get("usage", {})
            actual = usage.get("cost")
            tokens = usage.get("input_tokens", 0)
            if actual is not None:
                actual = float(actual)
                with self.db() as db:
                    db.execute(
                        "UPDATE calls SET status='settled', cost=?, tokens=? WHERE id=?", (actual, tokens, call_id)
                    )
                self.cost += actual - reserve
            answers = result["answers"]
            if set(answers) != set(questions):
                raise ValueError("Jev returned incomplete question answers")
            return answers
        except Exception as exc:
            archive.setdefault("error", str(exc))
            (self.directory / f"call_{call_id:07d}.json").write_text(json.dumps(archive, ensure_ascii=False))
            raise
