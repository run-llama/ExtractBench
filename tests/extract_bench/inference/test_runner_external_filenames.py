"""Non-LlamaCloud providers receive the document under a random basename.

The dataset filename (``acme_invoice_2024.pdf``) is visible to a hosted model and
to agent-style providers in their prompt, so it is replaced by a same-content
staged file with a neutral random name. The saved request keeps the benchmark
path so results still pair with their test case.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from extract_bench.inference.providers.base import Provider
from extract_bench.inference.runner import EXTERNAL_FILENAME_STEMS, InferenceRunner
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType


class _RecordingProvider(Provider):
    def __init__(self) -> None:
        self.seen_paths: list[str] = []

    @property
    def provider_name(self) -> str:  # type: ignore[override]
        return "recording"

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        self.seen_paths.append(request.source_file_path)
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=pipeline.product_type,
            raw_output={"content": Path(request.source_file_path).read_text()},
            started_at=datetime.now(),
            completed_at=datetime.now(),
            latency_in_ms=0,
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:  # pragma: no cover - unused here
        raise NotImplementedError


def _runner(provider_name: str, config: dict | None = None) -> tuple[InferenceRunner, _RecordingProvider]:
    runner = InferenceRunner.__new__(InferenceRunner)
    provider = _RecordingProvider()
    runner.provider = provider
    runner.pipeline = PipelineSpec(
        pipeline_name="p",
        provider_name=provider_name,
        product_type=ProductType.EXTRACT,
        config=config or {},
    )
    return runner, provider


def _request(path: Path) -> InferenceRequest:
    return InferenceRequest(example_id="g/doc", source_file_path=str(path), product_type=ProductType.EXTRACT)


def test_llamacloud_providers_keep_the_benchmark_filename() -> None:
    runner, _ = _runner("llamaextract_v2")
    assert runner._should_randomize_external_filename() is False


def test_external_providers_randomize_by_default_and_can_opt_out() -> None:
    assert _runner("openai_extract")[0]._should_randomize_external_filename() is True
    assert (
        _runner("openai_extract", {"randomize_external_filename": False})[0]._should_randomize_external_filename()
        is False
    )


def test_provider_sees_random_name_but_saved_request_keeps_source_path(tmp_path: Path) -> None:
    source = tmp_path / "acme_invoice_2024.pdf"
    source.write_text("same bytes")
    runner, provider = _runner("openai_extract")

    raw = runner._run_provider_inference(_request(source))

    assert len(provider.seen_paths) == 1
    seen = Path(provider.seen_paths[0])
    assert seen.suffix == ".pdf"
    assert seen.name != source.name
    assert "acme" not in seen.name
    assert any(stem in seen.name for stem in EXTERNAL_FILENAME_STEMS) or seen.stem.isalnum()
    assert raw.raw_output["content"] == "same bytes"
    assert raw.request.source_file_path == str(source)
    assert not seen.parent.exists(), "staged directory must be cleaned up"


def test_opted_out_pipeline_passes_the_original_path(tmp_path: Path) -> None:
    source = tmp_path / "acme_invoice_2024.pdf"
    source.write_text("x")
    runner, provider = _runner("openai_extract", {"randomize_external_filename": False})
    runner._run_provider_inference(_request(source))
    assert provider.seen_paths == [str(source)]
