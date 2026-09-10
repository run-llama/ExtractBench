"""Evaluator for EXTRACT product type using annotation-based evaluation."""

from typing import Any

from extract_bench.evaluation.evaluators.base import BaseEvaluator
from extract_bench.evaluation.grounded_confidence_payloads import confidence_field_rules
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
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.extract_field_paths import parse_field_path
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
        # Must be EXTRACT product type
        if inference_result.product_type != ProductType.EXTRACT:
            return False

        # Must have ExtractOutput
        if not isinstance(inference_result.output, ExtractOutput):
            return False

        # Must be ExtractTestCase
        if not isinstance(test_case, ExtractTestCase):
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
        if not self.can_evaluate(inference_result, test_case):
            raise ValueError("Cannot evaluate: missing expected_output or test_rules, or invalid product type")

        if not isinstance(inference_result.output, ExtractOutput):
            raise ValueError("Inference result output is not ExtractOutput")

        if not isinstance(test_case, ExtractTestCase):
            raise ValueError("Test case must be ExtractTestCase for EXTRACT evaluation")

        raw_extracted_data = inference_result.output.extracted_data
        metrics: list[MetricValue] = []
        diagnostic_metrics: list[MetricValue] = []

        # Normalize per_table_row list projections back into the per-doc shape
        # used by extract_field rules. The adapter is a pure shape transform:
        # skipped scalar paths are recorded on v0.2 evidence-metric metadata.
        field_rules_for_unwrap = (
            test_case.get_extract_field_rules() if hasattr(test_case, "get_extract_field_rules") else []
        )
        normalization = normalize_list_prediction(
            raw_extracted_data,
            field_rules_for_unwrap,
            data_schema=test_case.data_schema,
        )
        extracted_data = normalization.extracted_data
        unwrap_skipped = [
            *normalization.skipped_field_paths,
            *normalization.alias_skipped_field_paths,
        ]

        # Annotation-based evaluation.
        #
        # Note: the accuracy metric is computed against the *unwrapped*
        # extracted_data vs the full expected_output. On per_table_row runs
        # this honestly drops accuracy because scalar fields the prediction
        # doesn't emit (e.g. ``client_id``) still appear in expected_output.
        # That drop is a correct signal, not noise — if scalar coverage
        # matters, run a per_doc pipeline instead. See list_unwrap.py.
        # TODO: remove this legacy M0 path after the v0.2 migration drops
        # expected_output from all extract test cases.
        # Dataset-declared identity keys (match_by structural rules) let the
        # JSON subset match pair array rows by identity instead of by index,
        # so a reordered or dropped row doesn't cascade into every later row.
        identity_keys_by_path = _identity_keys_by_path(field_rules_for_unwrap)

        if test_case.expected_output is not None:
            expected_output = test_case.expected_output

            # Reconcile flat vs nested EOB claims layouts so the JSON subset
            # match metric does not score zero when only the nesting differs.
            extracted_data, expected_output = _normalize_eob_layouts(extracted_data, expected_output)

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
                )
            )

            # Unified evidence metric: array_record's keyless Hungarian alignment
            # (no match_by cascade) plus OR-acceptable evidence values and
            # page/bbox grounding. The *_value_* metrics equal array_record_*
            # when evidence adds no acceptable value; the *_grounded_* metrics
            # are 0 for pipelines that emit no citations.
            metrics.extend(
                compute_unified_evidence_metrics(
                    expected_output=expected_output,
                    extracted_data=extracted_data,
                    field_rules=test_case.get_extract_field_rules(),
                    field_citations=getattr(inference_result.output, "field_citations", []),
                    data_schema=test_case.data_schema,
                )
            )

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
                field_rules=test_case.get_extract_field_rules(),
                field_citations=getattr(inference_result.output, "field_citations", []),
                skip_field_paths=unwrap_skipped,
            )
        )
        metrics.extend(
            compute_confidence_scoped_metrics(
                extracted_data=extracted_data,
                field_rules=confidence_field_rules(test_case),
                field_citations=getattr(inference_result.output, "field_citations", []),
                # eval-side view: row identity re-attached from `_eval_row_identity`.
                # Alignment treats a schema-declared identity_key as authoritative,
                # unlike the gated `match_by:` rule path, so the block has to be
                # visible here even though it never ships to a provider.
                data_schema=test_case.eval_data_schema,
                skip_field_paths=set(unwrap_skipped),
                expected_output=getattr(test_case, "expected_output", None),
            )
        )

        stats = build_operational_stats(inference_result)

        return EvaluationResult(
            test_id=test_case.test_id,
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            product_type=inference_result.product_type.value,
            success=True,
            metrics=metrics,
            diagnostic_metrics=diagnostic_metrics,
            error=None,
            job_id=inference_result.raw_output.get("job_id"),
            parse_job_id=inference_result.raw_output.get("parse_job_id"),
            stats=stats,
        )

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

