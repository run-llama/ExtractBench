from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

from extract_bench.inference.providers.base import ProviderTransientError
from extract_bench.inference.providers.extract import codex_code_extract
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType

Provider = codex_code_extract.CodexCodeExtractProvider


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="codex_code_extract_gpt_5_4_low",
        provider_name="codex_code_extract",
        product_type=ProductType.EXTRACT,
        config={"model": "gpt-5.4", "reasoning_effort": "low"},
    )


def _request(source_file_path: str, example_id: str = "example-1") -> InferenceRequest:
    return InferenceRequest(
        example_id=example_id,
        source_file_path=source_file_path,
        product_type=ProductType.EXTRACT,
        schema_override={
            "type": "object",
            "properties": {"invoice_number": {"type": "string"}},
        },
    )


def _turn_completed(**usage_overrides: int) -> str:
    usage = {
        "input_tokens": 1000,
        "cached_input_tokens": 100,
        "output_tokens": 20,
        "reasoning_output_tokens": 5,
        **usage_overrides,
    }
    return json.dumps({"type": "turn.completed", "usage": usage})


class _TrackingLines:
    def __init__(self, lines: list[str]):
        self._lines = [line if line.endswith("\n") else line + "\n" for line in lines]
        self.consumed = 0

    def __iter__(self) -> _TrackingLines:
        return self

    def __next__(self) -> str:
        if self.consumed >= len(self._lines):
            raise StopIteration
        line = self._lines[self.consumed]
        self.consumed += 1
        return line


class _FakeStdin:
    def __init__(self) -> None:
        self.text = ""
        self.closed = False

    def write(self, text: str) -> int:
        self.text += text
        return len(text)

    def close(self) -> None:
        self.closed = True


class _FakePopen:
    def __init__(self, lines: list[str], *, returncode: int = 0):
        self.stdin = _FakeStdin()
        self.stdout = _TrackingLines(lines)
        self.returncode = returncode
        self._done = False
        self.terminate_called = False
        self.kill_called = False

    def wait(self, timeout: float | None = None) -> int:
        self._done = True
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode if (self._done or self.terminate_called) else None

    def terminate(self) -> None:
        self.terminate_called = True
        self._done = True

    def kill(self) -> None:
        self.kill_called = True
        self._done = True


def _png(tmp_path: Path) -> Path:
    source = tmp_path / "invoice.png"
    source.write_bytes(b"fake-image")
    return source


def test_build_cmd_has_expected_flags(tmp_path: Path) -> None:
    provider = Provider("codex_code_extract", {"model": "gpt-5.4", "reasoning_effort": "low"})
    cmd = provider._build_cmd(
        workdir=tmp_path,
        last_message_path=tmp_path / "last_message.txt",
    )

    assert cmd[0] == "codex"
    assert cmd[cmd.index("--ask-for-approval") + 1] == "never"
    assert cmd[cmd.index("--sandbox") + 1] == "workspace-write"
    assert cmd[cmd.index("--cd") + 1] == str(tmp_path)
    assert "--disable" in cmd
    assert "plugins" in cmd
    assert "exec" in cmd
    assert "--json" in cmd
    assert "--ephemeral" in cmd
    assert "--ignore-user-config" in cmd
    assert "--ignore-rules" in cmd
    assert cmd[cmd.index("--model") + 1] == "gpt-5.4"
    assert cmd[cmd.index("-c") + 1] == 'model_reasoning_effort="low"'
    assert "--output-schema" not in cmd
    assert cmd[cmd.index("--output-last-message") + 1] == str(tmp_path / "last_message.txt")
    assert cmd[-1] == "-"


def test_build_cmd_supports_danger_full_access_sandbox(tmp_path: Path) -> None:
    provider = Provider("codex_code_extract", {"model": "gpt-5.4", "sandbox": "danger-full-access"})
    cmd = provider._build_cmd(
        workdir=tmp_path,
        last_message_path=tmp_path / "last_message.txt",
    )

    assert cmd[cmd.index("--sandbox") + 1] == "danger-full-access"


def test_usage_from_events_sums_turns() -> None:
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "t"}),
        _turn_completed(input_tokens=10, cached_input_tokens=2, output_tokens=3, reasoning_output_tokens=1),
        "not-json",
        _turn_completed(input_tokens=7, cached_input_tokens=1, output_tokens=5, reasoning_output_tokens=2),
    ]

    assert Provider._usage_from_events(lines) == {
        "input_tokens": 17,
        "cached_input_tokens": 3,
        "output_tokens": 8,
        "reasoning_output_tokens": 3,
        "total_tokens": 25,
    }


def test_prepare_schema_preserves_required_fields_by_default() -> None:
    provider = Provider("codex_code_extract", {})
    schema = provider._prepare_schema(
        {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": ["integer", "null"]},
                        },
                    },
                }
            },
        }
    )

    assert "required" not in schema
    assert schema["additionalProperties"] is False
    assert "required" not in schema["properties"]["rows"]["items"]
    assert schema["properties"]["rows"]["items"]["additionalProperties"] is False


def test_prepare_schema_can_require_nested_properties() -> None:
    provider = Provider("codex_code_extract", {"all_properties_required": True})
    schema = provider._prepare_schema(
        {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": ["integer", "null"]},
                        },
                    },
                }
            },
        }
    )

    assert schema["required"] == ["rows"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["rows"]["items"]["required"] == ["name", "value"]
    assert schema["properties"]["rows"]["items"]["additionalProperties"] is False


def test_estimate_cost_uses_cached_input_discount() -> None:
    provider = Provider("codex_code_extract", {"model": "gpt-5.4"})
    cost = provider._estimate_cost_usd(
        {
            "input_tokens": 1000,
            "cached_input_tokens": 100,
            "output_tokens": 20,
            "reasoning_output_tokens": 5,
            "total_tokens": 1020,
        }
    )

    expected = 900 / 1_000_000 * 2.50 + 100 / 1_000_000 * 0.25 + 20 / 1_000_000 * 15.00
    assert cost == pytest.approx(expected)


def test_estimate_cost_uses_gpt_5_5_rates() -> None:
    provider = Provider("codex_code_extract", {"model": "gpt-5.5"})
    cost = provider._estimate_cost_usd(
        {
            "input_tokens": 1000,
            "cached_input_tokens": 100,
            "output_tokens": 20,
            "reasoning_output_tokens": 5,
            "total_tokens": 1020,
        }
    )

    expected = 900 / 1_000_000 * 5.00 + 100 / 1_000_000 * 0.50 + 20 / 1_000_000 * 30.00
    assert cost == pytest.approx(expected)


def test_estimate_cost_uses_gpt_5_6_sol_rates() -> None:
    provider = Provider("codex_code_extract", {"model": "gpt-5.6-sol"})
    usage = {
        "input_tokens": 1000,
        "cached_input_tokens": 100,
        "output_tokens": 20,
        "reasoning_output_tokens": 5,
        "total_tokens": 1020,
    }

    cost = provider._estimate_cost_usd(usage)

    expected = 900 / 1_000_000 * 4.00 + 100 / 1_000_000 * 0.40 + 20 / 1_000_000 * 20.00
    assert cost == pytest.approx(expected)
    assert provider._uses_long_context_pricing("gpt-5.6-sol", {**usage, "input_tokens": 300_000})


def test_estimate_cost_applies_long_context_uplift() -> None:
    provider = Provider("codex_code_extract", {"model": "gpt-5.4"})
    usage = {
        "input_tokens": 300_000,
        "cached_input_tokens": 250_000,
        "output_tokens": 10_000,
        "reasoning_output_tokens": 1_000,
        "total_tokens": 310_000,
    }

    cost = provider._estimate_cost_usd(usage)

    expected = 50_000 / 1_000_000 * 5.00 + 250_000 / 1_000_000 * 0.50 + 10_000 / 1_000_000 * 22.50
    assert cost == pytest.approx(expected)
    assert provider._pricing_snapshot(usage) == {
        "pricing_basis": "openai_api_standard",
        "long_context_applied": True,
        "long_context_threshold_tokens": 272_000,
        "input_price_per_1m": 5.00,
        "cached_input_price_per_1m": 0.50,
        "output_price_per_1m": 22.50,
    }


def test_run_inference_reads_codex_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _FakePopen(
        [
            json.dumps({"type": "thread.started", "thread_id": "t"}),
            json.dumps(
                {"type": "item.completed", "item": {"type": "agent_message", "text": '{"invoice_number":"A1"}'}}
            ),
            _turn_completed(),
        ]
    )
    captured: dict[str, Any] = {}

    def factory(cmd: list[str], *args: Any, cwd: str | None = None, **kwargs: Any) -> _FakePopen:
        captured["env"] = kwargs["env"]
        assert cwd is not None
        output_path = Path(cwd) / "output.json"
        output_path.write_text(json.dumps({"invoice_number": "A1"}), encoding="utf-8")
        return fake

    monkeypatch.setattr(subprocess, "Popen", factory)
    provider = Provider("codex_code_extract", {"model": "gpt-5.4", "api_key": "sk-test"})

    raw = provider.run_inference(_pipeline(), _request(str(_png(tmp_path))))

    assert fake.stdin.closed
    assert "Extract structured data" in fake.stdin.text
    assert raw.raw_output["data"] == {"invoice_number": "A1"}
    assert raw.raw_output["usage"]["input_tokens"] == 1000
    assert raw.raw_output["cost_usd"] > 0
    assert raw.raw_output["pricing"]["long_context_applied"] is False
    assert raw.raw_output["_config"]["pricing_basis"] == "openai_api_standard"
    assert raw.raw_output["_config"]["sandbox"] == "workspace-write"
    assert raw.raw_output["_last_message_tail"] == ""
    assert captured["env"]["CODEX_API_KEY"] == "sk-test"


def test_run_inference_rejects_bwrap_sandbox_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _FakePopen(
        [
            json.dumps({"type": "thread.started", "thread_id": "t"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "/bin/sh -c true",
                        "exit_code": 1,
                        "status": "failed",
                        "aggregated_output": "bwrap: Creating new namespace failed: nesting depth exceeded (ENOSPC)\n",
                    },
                }
            ),
            json.dumps(
                {"type": "item.completed", "item": {"type": "agent_message", "text": '{"invoice_number":null}'}}
            ),
            _turn_completed(),
        ]
    )

    def factory(cmd: list[str], *args: Any, cwd: str | None = None, **kwargs: Any) -> _FakePopen:
        assert cwd is not None
        (Path(cwd) / "output.json").write_text(json.dumps({"invoice_number": None}), encoding="utf-8")
        return fake

    monkeypatch.setattr(subprocess, "Popen", factory)
    provider = Provider("codex_code_extract", {"model": "gpt-5.4"})

    with pytest.raises(ProviderTransientError, match="sandbox failed"):
        provider.run_inference(_pipeline(), _request(str(_png(tmp_path))))


class _SpanRecorder:
    def __init__(self, name: str, kwargs: dict[str, Any]):
        self.name = name
        self.kwargs = kwargs
        self.logged: list[dict[str, Any]] = []

    def __enter__(self) -> _SpanRecorder:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def log(self, **event: Any) -> None:
        self.logged.append(event)


class _RecordingTracer:
    enabled = True

    def __init__(self) -> None:
        self.spans: list[_SpanRecorder] = []
        self.flushed = 0

    def span(self, name: str, **kwargs: Any) -> _SpanRecorder:
        span = _SpanRecorder(name, kwargs)
        self.spans.append(span)
        return span

    def flush(self) -> None:
        self.flushed += 1


def test_every_registered_codex_pipeline_has_a_pricing_row() -> None:
    """An unpriced model reports $0.00 instead of failing, so guard the registry.

    ``_pricing_for_model`` returns zeros on a miss, which is indistinguishable
    from a genuinely free run in every downstream cost report. gpt-5.6-luna and
    gpt-5.6-terra shipped that way until this test existed.
    """
    from extract_bench.inference.pipelines import get_pipeline, list_pipelines

    unpriced = []
    for name in list_pipelines():
        spec = get_pipeline(name)
        if spec.provider_name != "codex_code_extract":
            continue
        model = spec.config.get("model", Provider.DEFAULT_MODEL)
        if Provider._pricing_for_model(model) == (0.0, 0.0, 0.0):
            unpriced.append(f"{name} (model={model})")

    assert not unpriced, "codex pipelines whose model has no pricing row (would report $0.00): " + ", ".join(unpriced)


def test_recompute_cost_reprices_a_saved_codex_run_from_recorded_usage() -> None:
    """A saved codex run picks up pricing-table corrections through the seam.

    Cost is estimated post-hoc from the usage event, so a run recorded under the
    old gpt-5.6-sol rate (5.00/0.50/30.00) keeps that figure until re-priced. The
    table now says 4.00/0.40/20.00; recompute_cost re-derives from usage.
    """
    provider = Provider("codex_code_extract", {"model": "gpt-5.6-sol"})
    usage = {
        "input_tokens": 20_000,
        "cached_input_tokens": 2_000,
        "output_tokens": 1_000,
        "reasoning_output_tokens": 100,
        "total_tokens": 21_000,
    }
    raw_output = {
        "data": {"invoice_number": "INV-001"},
        "usage": usage,
        "num_pages": 4,
        "cost_usd": (18_000 * 5.00 + 2_000 * 0.50 + 1_000 * 30.00) / 1_000_000,  # old table
        "cost_per_page_usd": 0.0,
    }

    provider.recompute_cost(raw_output)

    expected = provider._estimate_cost_usd(usage)
    assert expected == pytest.approx((18_000 * 4.00 + 2_000 * 0.40 + 1_000 * 20.00) / 1_000_000)
    assert raw_output["cost_usd"] == pytest.approx(expected)
    assert raw_output["cost_per_page_usd"] == pytest.approx(expected / 4)
    assert raw_output["pricing"]["input_price_per_1m"] == 4.00


def test_recompute_cost_leaves_codex_cost_alone_without_usage() -> None:
    """Re-pricing an artifact that predates usage accounting would zero a real cost."""
    provider = Provider("codex_code_extract", {"model": "gpt-5.6-sol"})
    raw_output = {"data": {}, "cost_usd": 1.23}
    provider.recompute_cost(raw_output)
    assert raw_output["cost_usd"] == 1.23
