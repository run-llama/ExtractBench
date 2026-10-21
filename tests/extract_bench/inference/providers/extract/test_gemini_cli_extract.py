"""Native request pricing and the child process's enforced gateway route."""

import json
import sys
from collections import deque
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("httpx")
pytest.importorskip("pypdf")


from extract_bench.inference.providers.extract.gemini_cli_extract import (
    GeminiCliExtractProvider,
    cost_from_request_usage,
)
from extract_bench.inference.providers.extract.gemini_cli_trace import GeminiCliTrace


@pytest.mark.parametrize(
    "model,usage,expected",
    [
        # 1000 input - 400 cached; 500 candidate + 143 hidden thinking tokens.
        (
            "gemini-4-argon",
            {
                "promptTokenCount": 1000,
                "cachedContentTokenCount": 400,
                "candidatesTokenCount": 500,
                "totalTokenCount": 1643,
            },
            0.00767,
        ),
        ("gemini-3.1-pro-preview-customtools", {"promptTokenCount": 200_000, "totalTokenCount": 200_100}, 0.4012),
        ("gemini-3.1-pro-preview-customtools", {"promptTokenCount": 200_001, "totalTokenCount": 200_101}, 0.801804),
        ("gemini-4-argon", {"promptTokenCount": 0, "totalTokenCount": 0}, 0),
        ("unpriced-helper", {"promptTokenCount": 10, "totalTokenCount": 20}, None),
        ("gemini-4-argon", {"promptTokenCount": 10}, None),
        ("gemini-4-argon", {"promptTokenCount": 10, "totalTokenCount": 5}, None),
        ("gemini-4-argon", {"promptTokenCount": True, "totalTokenCount": 5}, None),
        ("gemini-4-argon", {"promptTokenCount": 10, "totalTokenCount": 15, "cachedContentTokenCount": 11}, None),
        ("gemini-4-argon", {"promptTokenCount": 10, "totalTokenCount": 15, "toolUsePromptTokenCount": 5}, None),
    ],
)
def test_native_pricing_includes_thinking_and_context_tiers_and_preserves_unknown(model, usage, expected) -> None:
    """Visible output alone underprices thinking; absent usage and unknown rates must not become zero."""
    cost = cost_from_request_usage(model, usage, as_of=date(2026, 10, 21))
    assert cost is None if expected is None else cost == pytest.approx(expected)


def test_child_cannot_inherit_direct_google_credentials_or_bypass_recorder(tmp_path: Path, monkeypatch) -> None:
    """Personal API keys, endpoint overrides and custom headers must not leak into CLI/tool children."""
    for name in (
        "GOOGLE_API_KEY",
        "GOOGLE_GEMINI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_VERTEX_BASE_URL",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GEMINI_CLI_CUSTOM_HEADERS",
        "GEMINI_API_KEY_AUTH_MECHANISM",
    ):
        monkeypatch.setenv(name, "must-not-reach-child")
    monkeypatch.setenv("NO_PROXY", "example.test")
    monkeypatch.setenv("GEMINI_CLI_CUSTOM_HEADERS", "x-model-provider: must-not-reach-child")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://gateway.test/genai")
    provider = GeminiCliExtractProvider("gemini_cli_extract")
    home = tmp_path / "home"
    home.mkdir()
    lines = deque(maxlen=20)
    script = "import os,json,sys;sys.stdin.read();print(json.dumps(dict(os.environ)))"
    trace = GeminiCliTrace(tmp_path, {"example_id": "doc"})
    code, _ = provider._run_cli(
        [sys.executable, "-c", script],
        tmp_path,
        home,
        "doc",
        lines,
        "prompt",
        base_url="http://127.0.0.1:12345",
        trace=trace,
        tool_usage={"tool_counts": {}, "shell_commands": [], "failed_tool_calls": 0},
    )
    env = json.loads(lines[0])
    assert code == 0
    assert "must-not-reach-child" not in json.dumps(env)
    assert env["GOOGLE_GEMINI_BASE_URL"] == "http://127.0.0.1:12345"
    assert env["GEMINI_API_KEY"] == "local-recorder"
    assert "127.0.0.1" in env["NO_PROXY"] and "example.test" in env["NO_PROXY"]


@pytest.mark.parametrize("write_output", [True, False])
def test_public_runner_keeps_evidence_and_paid_usage_when_cli_output_is_missing(tmp_path, monkeypatch, write_output):
    """Exercise the public request/artifact boundary, subprocess protocol, normalizer and failure journal."""
    import httpx
    from pypdf import PdfWriter

    from extract_bench.inference.pipelines import get_pipeline
    from extract_bench.inference.providers.base import ProviderPermanentError
    from extract_bench.inference.providers.extract import gemini_cli_transport
    from extract_bench.inference.providers.registry import create_provider
    from extract_bench.inference.runner import InferenceRunner
    from extract_bench.schemas.pipeline_io import InferenceRequest
    from extract_bench.schemas.product import ProductType

    monkeypatch.setenv("GEMINI_API_KEY", "configured-test-key")
    monkeypatch.delenv("GOOGLE_GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://gateway.test/genai")
    monkeypatch.setenv("GEMINI_CLI_CUSTOM_HEADERS", "x-model-provider: gemini, authorization: gateway-secret")
    original_client = httpx.Client
    forwarded = []

    def upstream(request):
        forwarded.append(request)
        return httpx.Response(200, json={"usageMetadata": {"promptTokenCount": 1000, "totalTokenCount": 1200}})

    monkeypatch.setattr(
        gemini_cli_transport.httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(upstream), trust_env=False),
    )
    pipeline = get_pipeline("gemini_cli_extract_gemini_4_argon_medium_thinking_vision_evidence")
    provider = create_provider(pipeline)
    source = tmp_path / "doc.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(source)
    script = """
import json, os, sys, urllib.request
from pathlib import Path
sys.stdin.read()
url = os.environ["GOOGLE_GEMINI_BASE_URL"] + "/v1beta/models/gemini-4-argon:generateContent"
with urllib.request.urlopen(urllib.request.Request(url, data=b'{}')) as response:
    response.read()
if sys.argv[1] == "True":
    Path("output.json").write_text('{"name":"example"}')
Path("citations.json").write_text('[{"field_path":"name","page":1,"bbox":[0.1,0.2,0.3,0.4]}]')
print(json.dumps({"type":"result", "status":"success", "stats":{}}))
"""
    monkeypatch.setattr(provider, "_build_cmd", lambda: [sys.executable, "-c", script, str(write_output)])
    runner = InferenceRunner(provider, pipeline, tmp_path / "output", max_concurrent=1, use_rich=False)
    request = InferenceRequest(
        example_id="group/doc",
        source_file_path=str(source),
        product_type=ProductType.EXTRACT,
        schema_override={"type": "object", "properties": {"name": {"type": "string"}}},
    )
    try:
        if write_output:
            raw = runner._run_provider_inference(request)
            normalized = provider.normalize(raw)
            assert normalized.output.extracted_data == {"name": "example"}
            assert normalized.output.field_citations[0].field_path == "name"
            assert raw.request == request
            assert raw.raw_output["_cost_complete"] is True
        else:
            with pytest.raises(ProviderPermanentError, match="output.json was not written") as error:
                runner._run_provider_inference(request)
            assert Path(error.value.debug_payload["trace_path"]).exists()
        journal = next((tmp_path / "output" / "traces").glob("*.jsonl"))
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        assert events[-1]["event"] == "attempt.finished"
        assert events[-1]["outcome"] == ("success" if write_output else "error")
        assert events[-1]["cost_usd"] > 0
        assert events[-1]["cost_complete"] is True
        assert forwarded[0].headers["x-goog-api-key"] == "configured-test-key"
        assert forwarded[0].headers["authorization"] == "gateway-secret"
        assert forwarded[0].headers["x-model-provider"] == "gemini"
        assert "configured-test-key" not in journal.read_text()
        assert "gateway-secret" not in journal.read_text()
    finally:
        runner.shutdown()


def test_public_pipeline_matches_recorded_submission_prompt_settings_and_command(monkeypatch):
    """A provider change must not silently mislabel the frozen leaderboard submission's inference configuration."""
    from extract_bench.inference.pipelines import get_pipeline
    from extract_bench.inference.providers.registry import create_provider

    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    submission = Path(__file__).parents[5] / "submissions/gemini-4-argon"
    recorded = json.loads((submission / "config.json").read_text())
    provider = create_provider(get_pipeline(recorded["pipeline"]))
    schema = provider._prepare_schema({"type": "object", "properties": {"name": {"type": "string"}}})
    expected = (
        (submission / "prompt.txt")
        .read_text()
        .rstrip("\n")
        .replace("{prepared_schema_json}", json.dumps(schema, indent=2))
    )
    assert provider._build_prompt(schema, "input.pdf") == expected
    assert provider._build_cmd() == recorded["command"]
    assert provider._build_settings() == recorded["settings"]


def test_argon_introductory_pricing_expires_without_changing_historical_replay():
    usage = {"promptTokenCount": 1000, "cachedContentTokenCount": 400, "totalTokenCount": 1200}
    intro = cost_from_request_usage("gemini-4-argon", usage, as_of=date(2026, 11, 30))
    standard = cost_from_request_usage("gemini-4-argon", usage, as_of=date(2026, 12, 1))
    assert standard == pytest.approx(2 * intro)
