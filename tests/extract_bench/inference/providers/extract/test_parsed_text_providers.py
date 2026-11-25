"""Successful two-stage extraction should retain page markers and both charges."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("pypdf", reason="runners extra required")

from extract_bench.inference.providers.extract import deepseek_extract, glm_extract
from extract_bench.inference.providers.parse import llamaparse
from extract_bench.schemas.parse_output import PageIR, ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType


@pytest.mark.parametrize(
    ("module", "provider_class", "provider_name"),
    [
        (deepseek_extract, deepseek_extract.DeepSeekExtractProvider, "deepseek_extract"),
        (glm_extract, glm_extract.GLMExtractProvider, "glm_extract"),
    ],
    ids=["deepseek", "glm"],
)
def test_parsed_text_provider_sends_numbered_pages_and_preserves_both_stage_costs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: ModuleType,
    provider_class: type[deepseek_extract.DeepSeekExtractProvider | glm_extract.GLMExtractProvider],
    provider_name: str,
) -> None:
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"document fixture")
    pages = [PageIR(page_index=0, markdown="# Invoice"), PageIR(page_index=1, markdown="Total: 42")]
    parse = Mock()
    parse.run_inference.return_value = SimpleNamespace(raw_output={"cost_usd": 0.025})
    parse.normalize.return_value = SimpleNamespace(
        output=ParseOutput(
            task_type="parse",
            example_id="invoice",
            pipeline_name="llamaparse",
            pages=pages,
            markdown="# Invoice\n\nTotal: 42",
        )
    )
    monkeypatch.setattr(llamaparse, "LlamaParseProvider", Mock(return_value=parse))
    complete = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"total":42}'))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200),
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    monkeypatch.setattr(module, "OpenAI", Mock(return_value=client))
    config = {"api_key": "offline", "input_price_per_1m": 1.0, "output_price_per_1m": 2.0}
    provider = provider_class(provider_name, config)
    pipeline = PipelineSpec(pipeline_name=provider_name, provider_name=provider_name, product_type=ProductType.EXTRACT)
    request = InferenceRequest(
        example_id="invoice",
        source_file_path=str(source),
        product_type=ProductType.EXTRACT,
        schema_override={"type": "object", "properties": {"total": {"type": "number"}}},
    )

    raw = provider.run_inference(pipeline, request)

    complete.assert_called_once()
    call = complete.call_args.kwargs
    assert call["response_format"] == {"type": "json_object"}
    prompt = call["messages"][1]["content"]
    assert "<!-- Page 1 -->\n\n# Invoice\n\n<!-- Page 2 -->\n\nTotal: 42" in prompt
    assert '"total"' in prompt
    assert provider.normalize(raw).output.extracted_data == {"total": 42}
    assert raw.raw_output["num_pages"] == 2
    assert raw.raw_output["parse_cost_usd"] == pytest.approx(0.025)
    assert raw.raw_output["extract_cost_usd"] == pytest.approx(0.0014)
    assert raw.raw_output["cost_usd"] == pytest.approx(0.0264)
