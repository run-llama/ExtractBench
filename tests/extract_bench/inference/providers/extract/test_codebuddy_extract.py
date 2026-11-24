"""codebuddy_extract runs a local stand-in for the CodeBuddy CLI; no model is called."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required")

from extract_bench.inference.cost_contract import COST_USD, RAW_COST_COMPLETE
from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.agent_evidence import agent_evidence_instruction
from extract_bench.inference.providers.extract.codebuddy_extract import (
    CodeBuddyExtractProvider,
    usage_from_events,
)
from extract_bench.inference.providers.registry import create_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType

PIPELINE = "codebuddy_extract_hy4_flash_evidence"
_BASE_CONFIG = {
    "model": "hy4-flash",
    "url": "https://example.invalid/v1/chat/completions",
    "api_key": "test-key",
    "input_price_per_1m": 1.0,
    "output_price_per_1m": 2.0,
    "cache_read_price_per_1m": 0.1,
}


def _provider(**config: Any) -> CodeBuddyExtractProvider:
    return CodeBuddyExtractProvider("codebuddy_extract", {**_BASE_CONFIG, **config})


def _result_event(*, input_tokens: int = 1000, cache_read: int = 400, output_tokens: int = 50, **extra: Any) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "DONE",
            "usage": {
                "input_tokens": input_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": 0,
                "output_tokens": output_tokens,
            },
            **extra,
        }
    )


def _stage_event(stage_id: str, *, input_tokens: int, output_tokens: int, cache_read: int = 0) -> str:
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "cache_read_input_tokens": cache_read}
    return json.dumps(
        {
            "type": "system",
            "subtype": "usage_update",
            "_meta": {"codebuddy.ai/turnUsage": {"kind": "stage", "stage": {"id": stage_id, "usage": usage}}},
        }
    )


def test_hy4_flash_pipeline_uses_tokenhub_list_prices() -> None:
    spec = get_pipeline(PIPELINE)
    assert spec.provider_name == "codebuddy_extract"
    assert spec.product_type == ProductType.EXTRACT
    assert spec.config["model"] == "hy4-flash"
    assert spec.config["url"] == "https://tokenhub-intl.tencentcloudmaas.com/v1/chat/completions"
    assert spec.config["api_key_env"] == "TENCENT_TOKENHUB_API_KEY"
    assert (
        spec.config["input_price_per_1m"],
        spec.config["output_price_per_1m"],
        spec.config["cache_read_price_per_1m"],
    ) == (0.07, 0.281, 0.003)
    assert spec.config["enable_vision"] is True
    assert spec.config["evidence_mode"] is True
    assert spec.config["max_cost_usd"] is None
    assert isinstance(create_provider(spec), CodeBuddyExtractProvider)


@pytest.mark.parametrize("missing", ["model", "url", "input_price_per_1m"])
def test_missing_endpoint_model_or_prices_is_a_config_error(missing: str) -> None:
    config = {key: value for key, value in _BASE_CONFIG.items() if key != missing}
    with pytest.raises(ProviderConfigError, match=missing):
        CodeBuddyExtractProvider("codebuddy_extract", config)


def test_models_json_references_the_key_env_var_and_never_holds_the_key() -> None:
    config = _provider(api_key_env="TENCENT_TOKENHUB_API_KEY")._build_models_json()
    (entry,) = config["models"]
    assert entry["id"] == "hy4-flash"
    assert entry["url"] == "https://example.invalid/v1/chat/completions"
    assert entry["apiKey"] == "${TENCENT_TOKENHUB_API_KEY}"
    assert entry["supportsToolCall"] is True
    assert entry["supportsImages"] is True
    assert config["availableModels"] == ["hy4-flash"]
    assert "test-key" not in json.dumps(config)


def test_cmd_blocks_web_tools_and_host_settings() -> None:
    cmd = _provider()._build_cmd(Path("/tmp/mcp.json"))
    assert cmd[:3] == ["codebuddy", "-p", "-y"]
    assert cmd[cmd.index("--model") + 1] == "hy4-flash"
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert cmd[cmd.index("--disallowedTools") + 1] == "WebSearch,WebFetch"
    assert cmd[cmd.index("--setting-sources") + 1] == "user"
    assert "--strict-mcp-config" in cmd and "--no-session-persistence" in cmd


def test_prompt_adds_evidence_rule_and_moves_a_large_schema_to_a_file(tmp_path: Path) -> None:
    schema = {"type": "object", "properties": {"invoice_number": {"type": "string"}}}
    base = _provider()._build_prompt(schema, "input.pdf")
    assert (
        _provider(evidence_mode=True)._build_prompt(schema, "input.pdf") == base + "\n" + agent_evidence_instruction()
    )

    prompt, delivery = _provider()._stage_prompt(schema, "input.pdf", tmp_path)
    assert (delivery, prompt) == ("inline", base)
    big = {"type": "object", "properties": {f"field_{i:05d}": {"type": "string"} for i in range(5000)}}
    prompt, delivery = _provider()._stage_prompt(big, "input.pdf", tmp_path)
    assert delivery == "file"
    assert "./schema.json" in prompt and "field_00001" not in prompt
    assert json.loads((tmp_path / "schema.json").read_text()) == big


def test_usage_bills_cached_reads_once_and_dedupes_stages() -> None:
    events = [
        json.loads(_stage_event("model:m1", input_tokens=600, output_tokens=30)),
        json.loads(_stage_event("model:m2", input_tokens=400, output_tokens=20, cache_read=400)),
        json.loads(_result_event()),
    ]
    usage, observed, complete = usage_from_events(events)
    assert (observed, complete) == (True, True)
    assert usage["uncached_input_tokens"] == 600
    assert _provider()._estimate_cost(usage) == pytest.approx((600 * 1.0 + 400 * 0.1 + 50 * 2.0) / 1_000_000)

    # Without the terminal result event, repeated stage ids count once.
    partial, observed, complete = usage_from_events(
        [
            json.loads(_stage_event("model:m1", input_tokens=100, output_tokens=10)),
            json.loads(_stage_event("model:m1", input_tokens=600, output_tokens=30)),
        ]
    )
    assert (observed, complete) == (True, False)
    assert partial["input_tokens"] == 600
    assert usage_from_events([])[1] is False


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("429 Too Many Requests", ProviderRateLimitError),
        ("502 Bad Gateway", ProviderTransientError),
        ("402 The free trial quota for the service has been exhausted", ProviderPermanentError),
        ("model refused", ProviderPermanentError),
    ],
)
def test_error_results_are_classified(message: str, expected: type[Exception]) -> None:
    events = [{"type": "result", "is_error": True, "result": message}]
    with pytest.raises(expected):
        CodeBuddyExtractProvider._raise_for_status(1, events, "")


def _fake_codebuddy(tmp_path: Path, events: list[str], *, write_output: bool = True) -> Path:
    """A stand-in CodeBuddy CLI: prints its version, or records its inputs and writes outputs."""
    script = tmp_path / "codebuddy"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('2.185.0')\n"
        "    sys.exit(0)\n"
        "home = Path(os.environ['HOME'])\n"
        "models = home / '.codebuddy' / 'models.json'\n"
        "seen = {'argv': sys.argv[1:], 'prompt': sys.stdin.read(), 'cwd': os.getcwd(), 'home': str(home),\n"
        "        'models': json.loads(models.read_text()), 'mode': models.stat().st_mode & 0o777,\n"
        "        'key': os.environ.get('TENCENT_TOKENHUB_API_KEY'), 'pwd': os.environ.get('PWD'),\n"
        "        'cwd_entries': sorted(os.listdir('.'))}\n"
        "Path(__file__).with_suffix('.seen.json').write_text(json.dumps(seen))\n"
        f"if {write_output!r}:\n"
        "    Path('output.json').write_text(json.dumps({'invoice_number': 'A1'}))\n"
        "    Path('citations.json').write_text(json.dumps([{'field_path': 'invoice_number', 'page': 1,\n"
        "        'bbox': [0.1, 0.1, 0.2, 0.05]}]))\n"
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
        pipeline_name="codebuddy_test",
        provider_name="codebuddy_extract",
        product_type=ProductType.EXTRACT,
        config=config,
    )


def test_run_inference_reads_output_cost_citations_and_harness(tmp_path: Path) -> None:
    script = _fake_codebuddy(tmp_path, [json.dumps({"type": "system", "subtype": "init"}), _result_event()])
    config = {
        **_BASE_CONFIG,
        "api_key_env": "TENCENT_TOKENHUB_API_KEY",
        "codebuddy_bin": str(script),
        "harness_version": "2.185.0",
        "evidence_mode": True,
    }
    raw = CodeBuddyExtractProvider("codebuddy_extract", config).run_inference(_spec(config), _request(tmp_path))

    assert raw.raw_output["data"] == {"invoice_number": "A1"}
    assert raw.raw_output[COST_USD] == pytest.approx((600 * 1.0 + 400 * 0.1 + 50 * 2.0) / 1_000_000)
    assert raw.raw_output[RAW_COST_COMPLETE] is True
    assert raw.raw_output["harness"]["version"] == "2.185.0"
    assert raw.raw_output["schema_delivery"] == "inline"
    assert raw.raw_output["field_citations"][0]["field_path"] == "invoice_number"
    assert "test-key" not in json.dumps(raw.raw_output)

    seen = json.loads(script.with_suffix(".seen.json").read_text())
    assert "output.json" in seen["prompt"]
    assert seen["mode"] == 0o600
    assert seen["key"] == "test-key"
    assert seen["pwd"] == seen["cwd"]
    # The private HOME with the models file sits outside the agent's working directory.
    assert not Path(seen["home"]).is_relative_to(seen["cwd"])
    assert seen["cwd_entries"] == ["input.pdf", "schema.json"]
    assert seen["models"]["models"][0]["apiKey"] == "${TENCENT_TOKENHUB_API_KEY}"
    assert os.environ.get("HOME") != seen["home"]


def test_missing_output_json_fails_permanently_with_cost_attached(tmp_path: Path) -> None:
    script = _fake_codebuddy(tmp_path, [_result_event()], write_output=False)
    config = {**_BASE_CONFIG, "codebuddy_bin": str(script)}
    with pytest.raises(ProviderPermanentError, match="output.json was not written") as info:
        CodeBuddyExtractProvider("codebuddy_extract", config).run_inference(_spec(config), _request(tmp_path))
    assert info.value.debug_payload["cost_accounting"]["known_cost_usd"] == pytest.approx(
        (600 * 1.0 + 400 * 0.1 + 50 * 2.0) / 1_000_000
    )


def test_cost_limit_stops_the_cli_mid_run(tmp_path: Path) -> None:
    script = _fake_codebuddy(tmp_path, [_stage_event("model:m1", input_tokens=10_000_000, output_tokens=0)])
    config = {**_BASE_CONFIG, "codebuddy_bin": str(script), "max_cost_usd": 1.0}
    with pytest.raises(ProviderPermanentError, match="max_cost_usd"):
        CodeBuddyExtractProvider("codebuddy_extract", config).run_inference(_spec(config), _request(tmp_path))
