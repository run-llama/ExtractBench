"""The public extension surface and package metadata stay importable and consistent."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

import extract_bench
from extract_bench import extensions
from extract_bench.cli import BenchCLI
from extract_bench.evaluation.layout_adapters.registry import resolve_layout_provider_name
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    GradedCell,
    compute_unified_evidence_metrics,
)
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType, coerce_product_type
from extract_bench.test_cases.schema import ExtractFieldBbox, ExtractFieldTestRule


def test_extensions_exports_registration_hooks() -> None:
    for name in extensions.__all__:
        assert callable(getattr(extensions, name)), name
    assert {"register_provider", "register_pipeline"} <= set(extensions.__all__)


def test_version_is_the_single_source_of_truth() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", extract_bench.__version__)
    assert BenchCLI().version() == extract_bench.__version__
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in pyproject
    assert 'path = "src/extract_bench/__init__.py"' in pyproject


def test_package_is_typed() -> None:
    assert (Path(extract_bench.__file__).parent / "py.typed").exists()


# --- product types ----------------------------------------------------------


def test_register_product_type_accepted_in_pipeline_spec() -> None:
    qa = extensions.register_product_type("qa_ext")
    assert isinstance(qa, extensions.ExtensionProductType)
    assert qa == "qa_ext" and qa.value == "qa_ext" and qa.name == "QA_EXT"
    assert extensions.register_product_type("qa_ext") is qa  # idempotent
    assert "qa_ext" in extensions.registered_product_types()

    spec = PipelineSpec(pipeline_name="p", provider_name="x", product_type="qa_ext")
    assert spec.product_type == "qa_ext"
    assert spec.product_type.value == "qa_ext"
    assert PipelineSpec.model_validate_json(spec.model_dump_json()).product_type == qa

    builtin = PipelineSpec(pipeline_name="p", provider_name="x", product_type="extract")
    assert builtin.product_type is ProductType.EXTRACT
    assert builtin.product_type.value == "extract"


def test_unknown_product_type_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown product type"):
        coerce_product_type("definitely_not_registered")
    with pytest.raises(ValueError, match="built-in"):
        extensions.register_product_type("extract")


# --- output models ----------------------------------------------------------


class _QAOutput(BaseModel):
    task_type: Literal["qa_ext"] = "qa_ext"
    example_id: str
    answers: list[str] = Field(default_factory=list)


def _result(output: BaseModel | dict[str, Any]) -> InferenceResult:
    now = datetime.now()
    return InferenceResult(
        request=InferenceRequest(example_id="e", source_file_path="doc.pdf", product_type="extract"),
        pipeline_name="p",
        product_type="extract",
        raw_output={},
        output=output,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def test_register_output_model_round_trips_through_json() -> None:
    extensions.register_output_model("qa_ext", _QAOutput)
    assert extensions.registered_output_models()["qa_ext"] is _QAOutput
    result = _result({"task_type": "qa_ext", "example_id": "e", "answers": ["42"]})
    assert isinstance(result.output, _QAOutput)

    reloaded = InferenceResult.model_validate_json(result.model_dump_json())
    assert isinstance(reloaded.output, _QAOutput)
    assert reloaded.output.answers == ["42"]


def test_builtin_outputs_still_dispatch() -> None:
    result = _result({"task_type": "extract", "example_id": "e", "pipeline_name": "p", "extracted_data": {"a": 1}})
    assert isinstance(result.output, ExtractOutput)
    assert result.output.extracted_data == {"a": 1}
    with pytest.raises(ValueError, match="Unknown output task_type"):
        _result({"task_type": "nope", "example_id": "e"})
    with pytest.raises(ValueError, match="built-in"):
        extensions.register_output_model("extract", _QAOutput)


# --- pipeline resolver ------------------------------------------------------


def test_register_pipeline_resolver_wins_over_package_registry() -> None:
    spec = PipelineSpec(pipeline_name="harness_only_pipeline", provider_name="harness_provider", product_type="extract")

    def _resolver(pipeline_name: str) -> PipelineSpec | None:
        return spec if pipeline_name == "harness_only_pipeline" else None

    extensions.register_pipeline_resolver(_resolver)
    result = _result({"task_type": "extract", "example_id": "e", "pipeline_name": "harness_only_pipeline"})
    result.pipeline_name = "harness_only_pipeline"
    assert resolve_layout_provider_name(result) == "harness_provider"


# --- graded cell side channel ----------------------------------------------


def test_graded_cells_collects_verdicts_without_changing_scores() -> None:
    expected = {"invoice_number": "A-1"}
    actual = {"invoice_number": "A-1"}
    rules = [
        ExtractFieldTestRule(
            field_path="invoice_number",
            expected_value="A-1",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.2, 0.2])],
        )
    ]
    citations = [FieldCitation(field_path="invoice_number", page=1, bbox=[0.1, 0.1, 0.2, 0.2])]
    schema = {"type": "object", "properties": {"invoice_number": {"type": "string"}}}

    baseline = compute_unified_evidence_metrics(expected, actual, rules, citations, schema)
    cells: list[GradedCell] = []
    with_cells = compute_unified_evidence_metrics(expected, actual, rules, citations, schema, graded_cells=cells)

    assert [m.model_dump() for m in with_cells] == [m.model_dump() for m in baseline]
    assert len(cells) == 1
    cell = cells[0]
    assert cell.gt_path == "invoice_number"
    assert cell.value_correct and cell.page_correct and cell.grounded_correct
    assert cell.grounded_claim and cell.page_claim
