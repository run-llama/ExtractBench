"""Incremental local journals for Gemini CLI attempts, including failed requests."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class GeminiCliTrace:
    def __init__(self, directory: Path, identity: Mapping[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{uuid.uuid4().hex}.jsonl"
        self.cancelled = threading.Event()
        self._lock = threading.RLock()
        self._started = time.monotonic()
        self._finished = False
        self._accounting: dict[str, Any] = {"cost_usd": None, "cost_complete": False}
        self.emit("attempt.started", identity=identity)

    def emit(self, event: str, **payload: Any) -> None:
        # Opening per append also permits a cancelling HTTP worker to reconcile
        # late usage after the provider has returned. No finalizer can erase it.
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            row = {
                "event": event,
                "timestamp": datetime.now(UTC).isoformat(),
                "elapsed_s": time.monotonic() - self._started,
                **payload,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def record_cost(self, cost: float | None, *, complete: bool, usage: Mapping[str, Any]) -> None:
        with self._lock:
            self._accounting = {"cost_usd": cost, "cost_complete": complete, "usage": usage}
            self.emit("attempt.accounting", **self._accounting)

    def cancel(self, reason: str) -> None:
        with self._lock:
            if not self.cancelled.is_set():
                self.cancelled.set()
                self.emit("attempt.cancelled", reason=reason)

    def finish(self, outcome: str, **payload: Any) -> None:
        with self._lock:
            if not self._finished:
                self._finished = True
                self.emit("attempt.finished", outcome=outcome, **self._accounting, **payload)
