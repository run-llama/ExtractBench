import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from extract_bench.inference.providers.extract.jev.client import JevClient


def test_failure_reservation_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    client = JevClient(tmp_path, budget=0.0003)

    def fail(*args, **kwargs):
        raise TimeoutError("unknown outcome")

    monkeypatch.setattr(client.session, "post", fail)
    with pytest.raises(TimeoutError):
        client.decide("text", {"x": {"type": "noul", "instructions": "Is text present?"}})
    restarted = JevClient(tmp_path, budget=0.0003)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        restarted.decide("text", {"x": {"type": "noul", "instructions": "Is text present?"}})
    assert restarted.summary()["attempts"] == 1
    archive = json.loads(next(tmp_path.glob("call_*.json")).read_text())
    assert "Authorization" not in json.dumps(archive)


def test_settlement_and_missing_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    client = JevClient(tmp_path)

    class Response:
        status_code = 200

        def json(self):
            return {"answers": {}, "usage": {"cost": 0.00001, "input_tokens": 10}}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(client.session, "post", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="incomplete"):
        client.decide("text", {"x": {"type": "noul", "instructions": "Is text present?"}})
    assert client.summary()["accounted_usd"] == 0.00001


def test_concurrent_reservations_cannot_overspend(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    JevClient(tmp_path, budget=0.0004)

    def attempt(_):
        client = JevClient(tmp_path, budget=0.0004)

        def fail(*args, **kwargs):
            raise TimeoutError("unknown outcome")

        client.session.post = fail
        try:
            client.decide("text", {"x": {"type": "noul", "instructions": "Is text present?"}})
        except (TimeoutError, RuntimeError):
            pass

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(attempt, range(8)))
    summary = JevClient(tmp_path).summary()
    assert summary["accounted_usd"] <= 0.0004
    assert summary["attempts"] == 2
