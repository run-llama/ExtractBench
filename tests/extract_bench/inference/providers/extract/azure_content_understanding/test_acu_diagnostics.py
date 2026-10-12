import json

import pytest
import requests

from extract_bench.inference.providers.extract import azure_content_understanding as module
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType


def get_pipeline(name: str) -> PipelineSpec:
    return PipelineSpec(
        pipeline_name=name,
        provider_name="azure_content_understanding_extractbench"
        if name == "acu-extract"
        else "azure_content_understanding",
        product_type=ProductType.EXTRACT,
        config={
            "field_method": "generate" if name.endswith("gen") else "extract",
            "models": {
                "completion": "gpt-5.5" if name == "acu-extract" else "gpt-4.1",
                "embedding": "text-embedding-3-large",
            },
            "poll_interval": 3.0,
        },
    )


@pytest.mark.parametrize("pipeline_name", ["acu-ga-ext", "acu-ga-gen"])
@pytest.mark.parametrize(
    "diagnostics",
    [
        {},
        {"infos": None},
        {"infos": []},
        {
            "infos": [
                {"code": "LLMStats", "message": "completion calls: 2; total completion latency: 10.28s"},
                {"code": "FutureInfo", "message": "opaque"},
            ]
        },
    ],
)
def test_diagnostic_headers_survive_throttling_and_results(monkeypatch, tmp_path, pipeline_name, diagnostics):
    pipeline = get_pipeline(pipeline_name)
    provider = module.AzureContentUnderstandingProvider(
        "azure_content_understanding", {**pipeline.config, "endpoint": "https://example.test", "api_key": "offline"}
    )
    calls = []
    responses = [
        (202, {}, {"Operation-Location": "https://example.test/operation"}),
        (429, {}, {"Retry-After": "7"}),
        (200, {"status": "Running"}, {}),
        (200, {"status": "Succeeded", "result": {"contents": [{"kind": "document", "fields": {}}], **diagnostics}}, {}),
    ]

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        status, body, headers = responses.pop(0)
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(body).encode()
        response.headers.update(headers)
        return response

    monkeypatch.setattr(provider._session, "request", request)
    monkeypatch.setattr(provider, "_ensure_analyzer", lambda *a: ("fixture", {}, {}, "reused_ready"))
    sleeps = []
    original_request = module.content_understanding_request
    monkeypatch.setattr(
        module, "content_understanding_request", lambda *a, **kw: original_request(*a, **kw, sleep=sleeps.append)
    )
    intervals = []
    monkeypatch.setattr(
        module, "sleep_for_poll_interval", lambda deadline, interval: intervals.append(interval) or True
    )
    path = tmp_path / "fixture.pdf"
    path.write_bytes(b"offline pdf fixture")
    request_input = InferenceRequest(
        example_id="fixture",
        source_file_path=str(path),
        product_type=ProductType.EXTRACT,
        schema_override={"type": "object", "properties": {}},
    )
    raw = provider.run_inference(pipeline, request_input)
    assert [c[0] for c in calls] == ["POST", "GET", "GET", "GET"]
    assert all(c[2]["headers"]["x-ms-diagnostics"] == "true" for c in calls[1:])
    assert all(c[2]["headers"]["Ocp-Apim-Subscription-Key"] == "offline" for c in calls)
    assert sleeps == [7.0]
    assert intervals == [3.0]
    raw = RawInferenceResult.model_validate_json(raw.model_dump_json())
    normalized = InferenceResult.model_validate_json(provider.normalize(raw).model_dump_json())
    for result in (raw, normalized):
        saved = result.raw_output["result"]
        assert ("infos" in saved) == ("infos" in diagnostics)
        assert saved.get("infos") == diagnostics.get("infos")
