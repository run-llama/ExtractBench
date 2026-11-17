"""pi_extract runs a local stand-in for the pi CLI; no model is called."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required")

from extract_bench.inference.cost_contract import COMPLETE, COST_COMPONENT_EVENT, COST_USD, RAW_COST_COMPLETE
from extract_bench.inference.events import bind_provider_event_sink
from extract_bench.inference.limits import EXTRACT_HARD_TIMEOUT_S
from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.agent_evidence import agent_evidence_instruction
from extract_bench.inference.providers.extract.pi_extract import PiExtractProvider, summarize_pi_events
from extract_bench.inference.providers.registry import create_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType

PIPELINE = "pi_extract_hy4_vision_evidence"
_BASE_CONFIG = {
    "model": "hy4",
    "base_url": "https://example.invalid/v1",
    "input_price_per_1m": 1.0,
    "output_price_per_1m": 2.0,
    "cache_read_price_per_1m": 0.1,
}


def _provider(**config: Any) -> PiExtractProvider:
    return PiExtractProvider("pi_extract", {**_BASE_CONFIG, **config})


def _assistant_message_end(**usage_overrides: Any) -> str:
    usage = {
        "input": 1000,
        "cacheRead": 100,
        "cacheWrite": 0,
        "output": 20,
        "reasoning": 5,
        "totalTokens": 1020,
        "cost": {"input": 0.0005, "output": 0.0006, "cacheRead": 0.000005, "cacheWrite": 0, "total": 0.001105},
        **usage_overrides,
    }
    return json.dumps({"type": "message_end", "message": {"role": "assistant", "usage": usage}})


def _error_message_end(error_message: str) -> str:
    return json.dumps(
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "error", "errorMessage": error_message}}
    )


def test_hy4_pipeline_uses_tokenhub_list_prices_and_runner_deadline() -> None:
    spec = get_pipeline(PIPELINE)
    assert spec.provider_name == "pi_extract"
    assert spec.product_type == ProductType.EXTRACT
    assert spec.per_file_timeout is None
    assert not {"timeout", "timeout_s"} & set(spec.config)
    assert spec.config["base_url"] == "https://tokenhub-intl.tencentcloudmaas.com/v1"
    assert spec.config["api_key_env"] == "TENCENT_TOKENHUB_API_KEY"
    assert spec.config["sampling_params"] == {"reasoning_effort": "high"}
    assert spec.config["enable_vision"] is True
    assert spec.config["evidence_mode"] is True
    assert (
        spec.config["input_price_per_1m"],
        spec.config["output_price_per_1m"],
        spec.config["cache_read_price_per_1m"],
    ) == (0.834, 2.501, 0.042)
    assert EXTRACT_HARD_TIMEOUT_S == 3 * 60 * 60
    provider = create_provider(spec)
    assert isinstance(provider, PiExtractProvider)
    assert spec.config["max_cost_usd"] is None
    assert provider._max_cost_usd is None


@pytest.mark.parametrize("missing", ["model", "base_url", "input_price_per_1m"])
def test_missing_endpoint_model_or_prices_is_a_config_error(missing: str) -> None:
    config = {key: value for key, value in _BASE_CONFIG.items() if key != missing}
    with pytest.raises(ProviderConfigError, match=missing):
        PiExtractProvider("pi_extract", config)


def test_api_key_comes_from_the_named_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENCENT_TOKENHUB_API_KEY", "from-env")
    assert _provider(api_key_env="TENCENT_TOKENHUB_API_KEY")._api_key == "from-env"
    assert _provider(api_key_env="TENCENT_TOKENHUB_API_KEY", api_key="direct")._api_key == "direct"
    monkeypatch.delenv("PI_EXTRACT_API_KEY", raising=False)
    assert _provider()._api_key  # pi needs a non-empty key even without auth


def test_models_json_carries_endpoint_prices_sampling_and_vision() -> None:
    provider = _provider(
        provider_label="tencent-tokenhub",
        api_key="key",
        sampling_params={"reasoning_effort": "high"},
        enable_vision=True,
    )
    entry = provider._build_models_json()["providers"]["tencent-tokenhub"]
    assert entry["baseUrl"] == "https://example.invalid/v1"
    assert entry["api"] == "openai-completions"
    assert entry["apiKey"] == "key"
    (model,) = entry["models"]
    assert model["id"] == "hy4"
    assert model["input"] == ["text", "image"]
    assert model["samplingParams"] == {"reasoning_effort": "high"}
    assert model["cost"] == {"input": 1.0, "output": 2.0, "cacheRead": 0.1, "cacheWrite": 0}

    (plain,) = _provider()._build_models_json()["providers"]["openai-compatible"]["models"]
    assert plain["input"] == ["text"]
    assert "samplingParams" not in plain


def test_cmd_is_hermetic_and_keeps_the_prompt_out_of_argv() -> None:
    cmd = _provider(provider_label="tencent-tokenhub")._build_cmd()
    assert cmd[0] == "pi"
    assert cmd[cmd.index("--provider") + 1] == "tencent-tokenhub"
    assert cmd[cmd.index("--model") + 1] == "hy4"
    assert cmd[cmd.index("--mode") + 1] == "json"
    for flag in ("--approve", "--no-session", "--no-skills", "--no-extensions", "--no-prompt-templates"):
        assert flag in cmd
    assert cmd[-1] == "-p"


def test_prompt_adds_evidence_rule_and_extra_instructions() -> None:
    schema = {"type": "object", "properties": {"invoice_number": {"type": "string"}}}
    base = _provider()._build_prompt(schema, "input.pdf")
    evidence = _provider(evidence_mode=True)._build_prompt(schema, "input.pdf")
    assert evidence == base + "\n" + agent_evidence_instruction()
    assert _provider(extra_instructions="Note.")._build_prompt(schema, "input.pdf") == base + "\n\nNote."


def test_summarize_events_sums_assistant_turns_and_tool_failures() -> None:
    lines = [
        json.dumps({"type": "message_end", "message": {"role": "user"}}),
        _assistant_message_end(input=10, cacheRead=2, output=3, reasoning=1, totalTokens=13),
        "not-json",
        _assistant_message_end(input=7, cacheRead=1, output=5, reasoning=2, totalTokens=12),
        json.dumps({"type": "tool_execution_end", "toolName": "read", "isError": False}),
        json.dumps({"type": "tool_execution_end", "toolName": "bash", "isError": True, "args": {}, "result": "x"}),
        _error_message_end("404: model not found"),
    ]
    summary = summarize_pi_events(lines)
    assert summary.usage == {
        "input_tokens": 17,
        "cached_input_tokens": 3,
        "cache_write_tokens": 0,
        "output_tokens": 8,
        "reasoning_output_tokens": 3,
        "total_tokens": 25,
    }
    assert summary.cost["total"] == pytest.approx(0.001105 * 2)
    assert summary.error_messages == ["404: model not found"]
    assert summary.tool_diagnostics["completed"] == 2
    assert summary.tool_diagnostics["failures"] == [{"tool": "bash", "args": {}, "result": "x"}]


@pytest.mark.parametrize(
    ("error_message", "expected", "match"),
    [
        ("429 Too Many Requests: quota exceeded", ProviderRateLimitError, "rate-limited"),
        ("503 Service Unavailable: model overloaded", ProviderTransientError, "transient"),
        ("Provider finish_reason: content_filter", ProviderPermanentError, "CLI exited 0"),
    ],
)
def test_error_events_fail_even_when_pi_exits_zero(error_message: str, expected: type[Exception], match: str) -> None:
    summary = summarize_pi_events([_error_message_end(error_message)])
    with pytest.raises(expected, match=match):
        _provider()._raise_for_status(0, summary)


def _fake_pi(tmp_path: Path, events: list[str]) -> Path:
    """A stand-in pi CLI: prints its version, or records its inputs and writes outputs."""
    script = tmp_path / "pi"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('0.84.1')\n"
        "    sys.exit(0)\n"
        "home = Path(os.environ['PI_CODING_AGENT_DIR'])\n"
        "seen = {'argv': sys.argv[1:], 'prompt': sys.stdin.read(), 'cwd': os.getcwd(), 'home': str(home),\n"
        "        'models': json.loads((home / 'models.json').read_text()), 'cwd_entries': sorted(os.listdir('.'))}\n"
        "Path(__file__).with_suffix('.seen.json').write_text(json.dumps(seen))\n"
        "Path('output.json').write_text(json.dumps({'invoice_number': 'A1'}))\n"
        "Path('citations.json').write_text(json.dumps([{'field_path': 'invoice_number', 'page': 1,\n"
        "    'bbox': [0.1, 0.1, 0.2, 0.05]}]))\n"
        f"for line in {events!r}:\n"
        "    print(line)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _request(tmp_path: Path) -> InferenceRequest:
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"%PDF-1.4 fake")
    return InferenceRequest(
        example_id="invoice",
        source_file_path=str(source),
        product_type=ProductType.EXTRACT,
        schema_override={"type": "object", "properties": {"invoice_number": {"type": "string"}}},
    )


def _spec(config: dict[str, Any]) -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="pi_test", provider_name="pi_extract", product_type=ProductType.EXTRACT, config=config
    )


def test_run_inference_reads_output_cost_citations_and_harness(tmp_path: Path) -> None:
    script = _fake_pi(tmp_path, [json.dumps({"type": "session"}), _assistant_message_end()])
    config = {**_BASE_CONFIG, "pi_bin": str(script), "harness_version": "0.84.1", "evidence_mode": True}
    raw = PiExtractProvider("pi_extract", config).run_inference(_spec(config), _request(tmp_path))

    assert raw.raw_output["data"] == {"invoice_number": "A1"}
    assert raw.raw_output["cost_usd"] == pytest.approx(0.001105)
    assert raw.raw_output["usage"]["input_tokens"] == 1000
    assert raw.raw_output["harness"]["version"] == "0.84.1"
    assert raw.raw_output["_config"]["harness_version"] == "0.84.1"
    assert raw.raw_output["field_citations"][0]["field_path"] == "invoice_number"
    assert raw.raw_output["field_citations"][0]["page"] == 1

    seen = json.loads(script.with_suffix(".seen.json").read_text())
    assert "output.json" in seen["prompt"]
    assert not any("output.json" in arg for arg in seen["argv"])
    # The pi home with the key file sits outside the agent's working directory.
    assert not Path(seen["home"]).is_relative_to(seen["cwd"])
    assert seen["cwd_entries"] == ["input.pdf"]
    assert seen["models"]["providers"]["openai-compatible"]["baseUrl"] == "https://example.invalid/v1"


@pytest.mark.parametrize("prior_charge", [False, True])
def test_run_inference_marks_pi_cost_incomplete_when_a_turn_omits_it(tmp_path: Path, prior_charge: bool) -> None:
    missing = json.loads(_assistant_message_end())
    del missing["message"]["usage"]["cost"]
    lines = [_assistant_message_end()] if prior_charge else []
    script = _fake_pi(tmp_path, [*lines, json.dumps(missing)])
    config = {**_BASE_CONFIG, "pi_bin": str(script)}

    events: list[tuple[str, Mapping[str, Any]]] = []
    with bind_provider_event_sink(lambda name, payload: events.append((name, payload))):
        raw = PiExtractProvider("pi_extract", config).run_inference(_spec(config), _request(tmp_path)).raw_output

    assert raw[RAW_COST_COMPLETE] is False
    assert raw[COST_USD] == (pytest.approx(0.001105) if prior_charge else None)
    ((name, component),) = events
    assert name == COST_COMPONENT_EVENT
    assert component[COMPLETE] is False
    assert component[COST_USD] == raw[COST_USD]


def test_error_event_fails_with_trace_in_debug_payload(tmp_path: Path) -> None:
    script = _fake_pi(tmp_path, [_error_message_end("Provider finish_reason: content_filter")])
    config = {**_BASE_CONFIG, "pi_bin": str(script)}
    with pytest.raises(ProviderPermanentError, match="content_filter") as excinfo:
        PiExtractProvider("pi_extract", config).run_inference(_spec(config), _request(tmp_path))
    payload = excinfo.value.debug_payload
    assert payload["returncode"] == 0
    assert payload["output_json"] == {"invoice_number": "A1"}
    assert len(payload["trace_jsonl_lines"]) == 1


def test_harness_version_mismatch_stops_before_extraction(tmp_path: Path) -> None:
    script = _fake_pi(tmp_path, [_assistant_message_end()])
    config = {**_BASE_CONFIG, "pi_bin": str(script), "harness_version": "0.0.1"}
    with pytest.raises(ProviderConfigError, match="version mismatch"):
        PiExtractProvider("pi_extract", config).run_inference(_spec(config), _request(tmp_path))
    assert not script.with_suffix(".seen.json").exists()


def test_non_pdf_sources_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "invoice.png"
    source.write_bytes(b"image")
    request = _request(tmp_path).model_copy(update={"source_file_path": str(source)})
    with pytest.raises(ProviderPermanentError, match="PDFs only"):
        _provider().run_inference(_spec(_BASE_CONFIG), request)
