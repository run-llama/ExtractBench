"""The local layout matcher preserves upstream average-precision semantics."""

import pytest

from extract_bench.evaluation.metrics.layoutdet.classification_utils import compute_per_class_metrics


@pytest.mark.parametrize("with_page_ids", [False, True], ids=["flat", "per-page"])
def test_layout_ap_groups_equal_confidence_matches_and_misses(with_page_ids: bool) -> None:
    """Tied detections must share one threshold after flat or per-page box matching."""
    ground_truth = [
        {"bbox": [0, 0, 10, 10], "class_name": "Text"},
        {"bbox": [20, 0, 30, 10], "class_name": "Text"},
    ]
    predictions = [
        {**ground_truth[0], "score": 0.5},
        {**ground_truth[1], "score": 0.5},
        {"bbox": [40, 0, 50, 10], "class_name": "Text", "score": 0.5},
    ]
    if with_page_ids:
        ground_truth = [{**box, "example_id": "page-1"} for box in ground_truth]
        predictions = [{**box, "example_id": "page-1"} for box in predictions]

    metrics = compute_per_class_metrics(predictions, ground_truth, ["Text"])["Text"]

    assert metrics["ap"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0
