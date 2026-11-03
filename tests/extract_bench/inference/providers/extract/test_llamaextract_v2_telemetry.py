import json
from datetime import datetime
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

from extract_bench.inference.pipelines import get_pipeline  # noqa: E402
from extract_bench.inference.providers.extract import llamaextract_v2_api as module  # noqa: E402
from extract_bench.schemas.pipeline import PipelineSpec  # noqa: E402
from extract_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult  # noqa: E402
from extract_bench.schemas.product import ProductType  # noqa: E402


@pytest.mark.parametrize(
    "telemetry",
    [
        {"latency": {"parse_ms": 1000}, "metadata_state_transitions": [{"status": "COMPLETED"}]},
        {},
        {"latency": None, "metadata_state_transitions": None},
    ],
)
def test_poll_retains_optional_service_telemetry(monkeypatch, telemetry):
    provider = module.LlamaExtractV2Provider("llamaextract_v2", {"api_key": "offline"})
    calls = []
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    def handler(request):
        calls.append(request)
        assert request.url.params["project_id"] == "project"
        assert set(request.url.params.get_list("expand")) == {
            "metadata",
            "extract_metadata",
            "usage",
            "latency",
            "metadata_state_transitions",
        }
        status = "RUNNING" if len(calls) == 1 else "COMPLETED"
        return httpx.Response(
            200,
            json={
                "status": status,
                "extract_result": {"amount": 12},
                "metadata": {"usage": {"num_pages_extracted": 1}},
                "extract_metadata": {"parse_job_id": "parse-1"},
                **telemetry,
            },
        )

    with httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(handler)) as client:
        result = provider._poll_job(client, "job-1", {"project_id": "project"})
    assert len(calls) == 2
    assert result["latency"] == telemetry.get("latency")
    assert result["metadata_state_transitions"] == telemetry.get("metadata_state_transitions")
    assert result["metadata"]["usage"]["num_pages_extracted"] == 1
    assert result["spawned_parse_job_id"] == "parse-1"
    assert result["data"] == {"amount": 12}
    pipeline = PipelineSpec(
        pipeline_name="fixture", provider_name="llamaextract_v2", product_type=ProductType.EXTRACT, config={}
    )
    raw = RawInferenceResult(
        request=InferenceRequest(example_id="one", source_file_path="one.pdf", product_type=ProductType.EXTRACT),
        pipeline=pipeline,
        pipeline_name="fixture",
        product_type=ProductType.EXTRACT,
        raw_output=result,
        started_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1),
        latency_in_ms=0,
    )
    normalized = provider.normalize(raw)
    saved = normalized.model_dump(mode="json")
    assert saved["raw_output"]["latency"] == telemetry.get("latency")
    assert saved["raw_output"]["metadata_state_transitions"] == telemetry.get("metadata_state_transitions")


@pytest.mark.parametrize(
    ("billed_usage", "expected_cost"),
    [
        ({"extract_credits": 800, "parse_credits": 20}, 1.025),
        (None, None),
        ({"extract_credits": 800, "parse_credits": None}, None),
        ({"extract_credits": 0, "parse_credits": 0}, 0.0),
    ],
)
def test_registered_max_request_preserves_billed_model_cost_or_unknown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    billed_usage: dict[str, int | None] | None,
    expected_cost: float | None,
) -> None:
    """Dropping effort or billed usage must not silently price Max as ordinary Plus."""
    pipeline = get_pipeline("llamaextract_agentic_plus_max_effort")
    source = tmp_path / "document.txt"
    source.write_text("A synthetic invoice", encoding="utf-8")
    submitted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/beta/files":
            return httpx.Response(200, json={"id": "file-1"})
        if request.method == "POST" and request.url.path == "/api/v2/extract":
            submitted.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "job-1"})
        assert request.url.path == "/api/v2/extract/job-1"
        usage = billed_usage if "usage" in request.url.params.get_list("expand") else None
        return httpx.Response(
            200,
            json={
                "status": "COMPLETED",
                "extract_result": {"amount": 12},
                "metadata": {"usage": {"num_pages_extracted": 2, "num_pages_billed": 10}},
                "usage": usage,
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        module.httpx, "Client", lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handler))
    )
    provider = module.LlamaExtractV2Provider("llamaextract_v2", {"api_key": "offline", **pipeline.config})
    request = InferenceRequest(
        example_id="one",
        source_file_path=str(source),
        product_type=ProductType.EXTRACT,
        schema_override={"type": "object", "properties": {"amount": {"type": "number"}}},
    )
    raw = provider.run_inference(pipeline, request)
    assert submitted[0]["configuration"]["tier"] == "agentic_plus"
    assert submitted[0]["configuration"]["effort"] == "max"
    normalized = provider.normalize(raw)
    assert normalized.raw_output["usage"] == billed_usage
    assert normalized.raw_output.get("cost_usd") == expected_cost

    # A later read without settled usage must clear an earlier derived charge.
    raw.raw_output["usage"] = None
    repriced = provider.normalize(raw)
    assert repriced.raw_output.get("cost_usd") is None
    assert repriced.raw_output.get("credits_used") is None
