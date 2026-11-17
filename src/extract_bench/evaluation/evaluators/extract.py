"""Evaluator for EXTRACT product type using annotation-based evaluation.

Metric computation is deliberately separate from result construction. A
downstream harness that keeps its own ``EvaluationResult`` model (with its own
provenance fields) calls :meth:`ExtractEvaluator.compute_metrics` to get this
package's metric selection, adds its own metrics, and builds its own result —
without either side having to copy the other's metric wiring.
"""

from collections.abc import MutableSequence
from dataclasses import dataclass, field
from typing import Any, TypeGuard

from extract_bench.evaluation.evaluators.base import BaseEvaluator
from extract_bench.evaluation.metrics.extract.array_record_match_metric import (
    ArrayRecordMatchMetric,
)
from extract_bench.evaluation.metrics.extract.confidence_scoped.summary import (
    compute_confidence_scoped_metrics,
)
from extract_bench.evaluation.metrics.extract.json_subset_match_metric import (
    JsonSubsetMatchMetric,
)
from extract_bench.evaluation.metrics.extract.list_unwrap import normalize_list_prediction
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    ValueCell,
    compute_unified_evidence_metrics,
)
from extract_bench.evaluation.metrics.field_grounding.evidence_comparator import (
    parse_match_by_keys,
)
from extract_bench.evaluation.metrics.field_grounding.extract_adapter import (
    compute_extract_field_grounding_metrics,
)
from extract_bench.evaluation.stats import build_operational_stats
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.extract_field_paths import parse_field_path
from extract_bench.test_cases.identity_hints import top_level_identity_fields
from extract_bench.test_cases.schema import ExtractFieldTestRule, ExtractTestCase, TestCase


def _flatten_pd_claims(d: Any) -> Any:
    """Pull `claims` out of every payment_details[*] up to the doc root."""
    if not isinstance(d, dict):
        return d
    pd = d.get("payment_details")
    if not isinstance(pd, list):
        return d
    has_nested = any(isinstance(p, dict) and isinstance(p.get("claims"), list) for p in pd)
    if not has_nested:
        return d
    out = dict(d)
    flat: list[Any] = []
    new_pd: list[Any] = []
    for p in pd:
        if isinstance(p, dict) and isinstance(p.get("claims"), list):
            flat.extend(p["claims"])
            new_pd.append({k: v for k, v in p.items() if k != "claims"})
        else:
            new_pd.append(p)
    out["payment_details"] = new_pd
    existing = out.get("claims") or []
    out["claims"] = list(existing) + flat
    return out


def _normalize_eob_layouts(extract: Any, gt: Any) -> tuple[Any, Any]:
    """Reconcile flat vs nested claims layouts before scoring.

    EOB GTs sometimes nest claims under payment_details[*].claims while
    extracts emit a flat top-level claims list (or vice versa). Without
    reconciliation the JsonSubsetMatchMetric pairs an empty list against
    a populated one and scores near-zero. We flatten the nested side so
    both layouts look the same. No-op when shapes already agree.

    Also reconciles payment_details cardinality: singleton-schema variants
    declare `payment_details` as a single object while GT is authored as
    an array. The subset-match metric short-circuits to 0.0 on a
    dict-vs-list mismatch, so we wrap the singular side as a 1-element
    list before scoring.
    """
    if isinstance(extract, dict) and isinstance(gt, dict):
        ext_pd = extract.get("payment_details")
        gt_pd = gt.get("payment_details")
        if isinstance(ext_pd, dict) and isinstance(gt_pd, list):
            extract = {**extract, "payment_details": [ext_pd]}
        elif isinstance(gt_pd, dict) and isinstance(ext_pd, list):
            gt = {**gt, "payment_details": [gt_pd]}
    ext_top = isinstance(extract, dict) and isinstance(extract.get("claims"), list) and extract["claims"]
    gt_top = isinstance(gt, dict) and isinstance(gt.get("claims"), list) and gt["claims"]
    ext_nested = (
        isinstance(extract, dict)
        and isinstance(extract.get("payment_details"), list)
        and any(isinstance(p, dict) and isinstance(p.get("claims"), list) for p in extract["payment_details"])
    )
    gt_nested = (
        isinstance(gt, dict)
        and isinstance(gt.get("payment_details"), list)
        and any(isinstance(p, dict) and isinstance(p.get("claims"), list) for p in gt["payment_details"])
    )
    if ext_top and gt_nested and not gt_top:
        gt = _flatten_pd_claims(gt)
    elif gt_top and ext_nested and not ext_top:
        extract = _flatten_pd_claims(extract)
    return extract, gt


def is_extract_test_case(test_case: Any) -> TypeGuard[ExtractTestCase]:
    """Whether ``test_case`` carries extract ground truth this evaluator can read.

    Structural rather than an ``isinstance``: a downstream harness has its own
    test-case hierarchy (its ``ExtractTestCase`` extends its own base, not this
    one), and rejecting it would make the extract metrics unreachable from the
    harness this package is meant to be embedded in. A parse or layout case
    still fails the check, so evaluator routing is unaffected.
    """
    if isinstance(test_case, ExtractTestCase):
        return True
    return callable(getattr(test_case, "get_extract_field_rules", None)) and hasattr(test_case, "data_schema")


@dataclass(frozen=True)
class ExtractScoringInputs:
    """The normalized view of one extract result that every metric scores.

    Handed back with the metrics so a caller adding its own metrics scores the
    same ``extracted_data`` the package did — list-unwrap normalization and EOB
    layout reconciliation already applied — instead of re-deriving it and
    drifting from the headline numbers.
    """

    test_case: ExtractTestCase
    extracted_data: Any
    expected_output: dict[str, Any] | None
    field_citations: list[FieldCitation]
    identity_keys_by_path: dict[tuple[str, ...], list[str]]
    skip_field_paths: list[str]
    field_rules: list[ExtractFieldTestRule]

    def compute_unified_metrics(
        self,
        value_cells: MutableSequence[ValueCell] | None,
    ) -> list[MetricValue]:
        return compute_unified_evidence_metrics(
            expected_output=self.expected_output,
            extracted_data=self.extracted_data,
            field_rules=self.field_rules,
            field_citations=self.field_citations,
            data_schema=self.test_case.data_schema,
            value_cells=value_cells,
        )


def prepare_extract_scoring_inputs(
    test_case: ExtractTestCase,
    extracted_data: Any,
    field_citations: list[FieldCitation],
) -> ExtractScoringInputs:
    field_rules = test_case.get_extract_field_rules()
    normalization = normalize_list_prediction(
        extracted_data,
        field_rules,
        data_schema=test_case.data_schema,
    )
    extracted_data, expected_output = _normalize_eob_layouts(normalization.extracted_data, test_case.expected_output)
    return ExtractScoringInputs(
        test_case=test_case,
        extracted_data=extracted_data,
        expected_output=expected_output,
        field_citations=field_citations,
        identity_keys_by_path=_identity_keys_by_path(field_rules),
        skip_field_paths=[*normalization.skipped_field_paths, *normalization.alias_skipped_field_paths],
        field_rules=field_rules,
    )


@dataclass
class ExtractMetricBundle:
    """Metrics for one extract result, split by how reports consume them."""

    metrics: list[MetricValue] = field(default_factory=list)
    diagnostic_metrics: list[MetricValue] = field(default_factory=list)
    inputs: ExtractScoringInputs | None = None


class ExtractEvaluator(BaseEvaluator):
    """Evaluator for EXTRACT product type."""

    def __init__(
        self,
        case_sensitive: bool = False,
        cosine_similarity: bool = False,
        normalize_dates: bool = True,
        weighted: bool = True,
    ):
        """
        Initialize the extract evaluator.

        :param case_sensitive: Whether string comparison should be case-sensitive
        :param cosine_similarity: Use embedding similarity for strings (requires OpenAI API key)
        :param normalize_dates: Normalize date strings before comparison
        """
        self._accuracy_metric = JsonSubsetMatchMetric(
            case_sensitive=case_sensitive,
            cosine_similarity=cosine_similarity,
            normalize_dates=normalize_dates,
            weighted=weighted,
        )
        self._array_record_metric = ArrayRecordMatchMetric(normalize_dates=normalize_dates)

    def can_evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> bool:
        """
        Check if this evaluator can evaluate the given inference result and test case.

        :param inference_result: The inference result to evaluate
        :param test_case: The test case to evaluate against
        :return: True if this evaluator can handle this case
        """
        # Must be EXTRACT product type. Compared by value: a harness may carry
        # its own product-type enum whose EXTRACT member is a distinct object.
        product_type = getattr(inference_result.product_type, "value", inference_result.product_type)
        if product_type != ProductType.EXTRACT.value:
            return False

        # Must have ExtractOutput
        if not isinstance(inference_result.output, ExtractOutput):
            return False

        # Must carry extract ground truth (a harness's own case class counts)
        if not is_extract_test_case(test_case):
            return False

        # Need expected_output or extract_field rules
        has_expected_output = test_case.expected_output is not None
        has_test_rules = test_case.test_rules is not None and len(test_case.test_rules) > 0

        return has_expected_output or has_test_rules

    def evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> EvaluationResult:
        """
        Evaluate an EXTRACT inference result against a test case.

        :param inference_result: The inference result to evaluate
        :param test_case: The test case with expected output or field rules
        :return: Evaluation result with accuracy metrics
        :raises ValueError: If neither expected_output nor test_rules are provided
        """
        bundle = self.compute_metrics(inference_result, test_case)
        return EvaluationResult(
            test_id=test_case.test_id,
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            product_type=getattr(inference_result.product_type, "value", inference_result.product_type),
            success=True,
            metrics=bundle.metrics,
            diagnostic_metrics=bundle.diagnostic_metrics,
            error=None,
            job_id=inference_result.raw_output.get("job_id"),
            parse_job_id=inference_result.raw_output.get("parse_job_id"),
            stats=build_operational_stats(inference_result),
        )

    def compute_metrics(
        self,
        inference_result: InferenceResult,
        test_case: TestCase,
        *,
        value_cells: MutableSequence[ValueCell] | None = None,
    ) -> ExtractMetricBundle:
        """
        Compute this package's extract metrics, without building a result model.

        :param inference_result: The inference result to score
        :param test_case: The test case with expected output or field rules
        :return: Headline and diagnostic metrics, plus the normalized inputs they scored
        :raises ValueError: If neither expected_output nor test_rules are provided
        """
        if not self.can_evaluate(inference_result, test_case):
            raise ValueError("Cannot evaluate: missing expected_output or test_rules, or invalid product type")

        if not isinstance(inference_result.output, ExtractOutput):
            raise ValueError("Inference result output is not ExtractOutput")

        if not is_extract_test_case(test_case):
            raise ValueError("Test case must carry extract ground truth for EXTRACT evaluation")

        inputs = prepare_extract_scoring_inputs(
            test_case,
            inference_result.output.extracted_data,
            list(inference_result.output.field_citations),
        )
        extracted_data = inputs.extracted_data
        expected_output = inputs.expected_output
        identity_keys_by_path = inputs.identity_keys_by_path
        metrics: list[MetricValue] = []
        diagnostic_metrics: list[MetricValue] = []
        scored_cells: list[ValueCell] = []

        if expected_output is not None:
            # Calculate overall accuracy using the metric
            accuracy_metric = self._accuracy_metric.compute(
                expected=expected_output,
                actual=extracted_data,
                identity_keys_by_path=identity_keys_by_path,
                data_schema=test_case.data_schema,
            )
            metrics.append(accuracy_metric)

            metrics.extend(
                self._array_record_metric.compute(
                    expected=expected_output,
                    actual=extracted_data,
                    data_schema=test_case.data_schema,
                    identity_keys_by_field=top_level_identity_fields(getattr(test_case, "identity_hints", None)),
                )
            )

            metrics.extend(inputs.compute_unified_metrics(scored_cells))
            if value_cells is not None:
                value_cells.extend(scored_cells)

            # Calculate field-level accuracy if both are dicts
            if isinstance(expected_output, dict) and isinstance(extracted_data, dict):
                schema_props = (
                    test_case.data_schema.get("properties") if isinstance(test_case.data_schema, dict) else None
                )
                for key in expected_output.keys():
                    expected_value = expected_output.get(key)
                    actual_value = extracted_data.get(key)
                    field_schema = (
                        schema_props.get(key) if isinstance(schema_props, dict) and key in schema_props else None
                    )
                    field_result = self._accuracy_metric.compute(
                        expected=expected_value,
                        actual=actual_value,
                        identity_keys_by_path={
                            path[1:]: keys for path, keys in identity_keys_by_path.items() if path and path[0] == key
                        },
                        data_schema=field_schema,
                    )
                    diagnostic_metrics.append(
                        MetricValue(
                            metric_name=f"field_accuracy_{key}",
                            value=field_result.value,
                            metadata={"field": key, **field_result.metadata},
                        )
                    )

        metrics.extend(
            compute_extract_field_grounding_metrics(
                extracted_data=extracted_data,
                field_rules=inputs.field_rules,
                field_citations=inputs.field_citations,
                skip_field_paths=inputs.skip_field_paths,
            )
        )
        metrics.extend(
            compute_confidence_scoped_metrics(
                extracted_data=extracted_data,
                expected_output=expected_output,
                field_rules=inputs.field_rules,
                field_citations=inputs.field_citations,
                data_schema=test_case.data_schema,
                value_cells=scored_cells,
            )
        )

        return ExtractMetricBundle(metrics=metrics, diagnostic_metrics=diagnostic_metrics, inputs=inputs)


def _identity_keys_by_path(field_rules: list[ExtractFieldTestRule]) -> dict[tuple[str, ...], list[str]]:
    """Dataset-declared identity keys per array path, from ``match_by`` rules.

    Keys are dict-key path tuples (``("line_items",)``); array parents whose
    own path contains a list index are skipped — a per-row identity scope is
    ambiguous for the path-based pairing in the JSON subset match.
    """
    identity_keys: dict[tuple[str, ...], list[str]] = {}
    for rule in field_rules:
        structural = rule.structural or ""
        if not structural.startswith("match_by:"):
            continue
        keys = parse_match_by_keys(structural.split(":", 1)[1])
        if not keys:
            continue
        try:
            tokens = parse_field_path(rule.field_path)
        except ValueError:
            continue
        if any(isinstance(token, int) for token in tokens):
            continue
        identity_keys[tuple(str(token) for token in tokens)] = keys
    return identity_keys
