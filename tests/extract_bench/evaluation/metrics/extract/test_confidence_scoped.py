"""Confidence summaries score returned scalar fields, not their containers."""

from extract_bench.evaluation.metrics.extract.confidence_scoped.summary import compute_confidence_scoped_metrics


def test_parent_citations_do_not_change_scalar_confidence_metrics() -> None:
    extracted = {
        "record": {"amount": 9, "note": None},
        "active": False,
        "labels": ["A", "B"],
        "empty_object": {},
        "empty_array": [],
    }
    expected = {**extracted, "record": {"amount": 10, "note": None}}
    citations = [
        {"field_path": path, "confidence": score}
        for path, score in [
            ("record.amount", 0.25),
            ("record.note", 1.0),
            ("active", 1.0),
            ("labels[0]", 1.0),
            ("labels[1]", 1.0),
        ]
    ]
    baseline = compute_confidence_scoped_metrics(
        extracted_data=extracted, expected_output=expected, field_rules=[], field_citations=citations
    )
    with_parents = compute_confidence_scoped_metrics(
        extracted_data=extracted,
        expected_output=expected,
        field_rules=[],
        field_citations=citations
        + [{"field_path": path, "confidence": 1.0} for path in ["record", "labels", "empty_object", "empty_array"]],
    )

    assert with_parents == baseline
    carrier = next(metric for metric in with_parents if metric.metric_name == "confidence_scoped_auc")
    assert carrier.metadata["total_fields"] == 5
    assert carrier.metadata["correct_fields"] == 4
    assert {row["predicted_path"] for row in carrier.metadata["field_rows_sample"]} == {
        citation["field_path"] for citation in citations
    }
    threshold = next(row for row in carrier.metadata["thresholds"] if row["threshold"] == 0.8)
    assert threshold["selected"] == threshold["correct_selected"] == 4


def test_parent_only_confidence_does_not_score_uncited_children() -> None:
    extracted = {"record": {"amount": 10}}
    metrics = compute_confidence_scoped_metrics(
        extracted_data=extracted,
        expected_output=extracted,
        field_rules=[],
        field_citations=[{"field_path": "record", "confidence": 1.0}],
    )
    carrier = next(metric for metric in metrics if metric.metric_name == "confidence_scoped_auc")
    coverage = next(metric for metric in metrics if metric.metric_name == "confidence_signal_coverage")
    assert carrier.metadata["total_fields"] == 0
    assert carrier.metadata["field_rows_sample"] == []
    assert coverage.value == 0


def test_sparse_alignment_matches_repeated_identical_rows_once_each() -> None:
    # Frequent redacted rows have no rare feature for sparse candidate lookup.
    # Two distinct anchors make the partial alignment nonempty, exposing misses.
    repeated = {"name": "NAME ON FILE", "address": "ADDRESS ON FILE"}
    anchors = [{"name": "ALPHA", "address": "1 A STREET"}, {"name": "BETA", "address": "2 B STREET"}]
    expected = {"rows": [repeated] * 201 + anchors}
    extracted = {"rows": anchors[::-1] + [{"address": "ADDRESS ON FILE", "name": "NAME ON FILE"}] * 202}
    metrics = compute_confidence_scoped_metrics(
        extracted_data=extracted,
        expected_output=expected,
        field_rules=[],
        field_citations=[
            {"field_path": f"rows[{index}].{key}", "confidence": 1.0}
            for index in range(len(extracted["rows"]))
            for key in ("name", "address")
        ],
    )
    carrier = next(metric for metric in metrics if metric.metric_name == "confidence_scoped_auc")
    assert carrier.metadata["total_fields"] == 408
    assert carrier.metadata["correct_fields"] == 406
    assert carrier.metadata["full_gt"]["unique_leaf_covered"] == 406
