from __future__ import annotations

import math
import random
import struct
from collections.abc import Sequence

import pytest
from _python_geometry_reference import _rect_union_area as reference_union
from _python_geometry_reference import compute_standard_iou_metrics as reference_iou

from extract_bench._native import rect_union_areas
from extract_bench.evaluation.metrics.field_grounding.core import (
    BBox,
    _rect_union_area,
    _rect_union_areas,
    compute_standard_iou_metrics,
    field_grounding_localization_passes,
)


def assert_same_float(actual: float, expected: float) -> None:
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert struct.pack("!d", actual) == struct.pack("!d", expected)


@pytest.mark.parametrize(
    "rectangles",
    [
        [],
        [(0.0, 0.0, 1.0, 1.0)],
        [(0.0, 0.0, 2.0, 2.0)] * 3 + [(1.0, 1.0, 3.0, 3.0)],
        [(0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 2.0, 1.0)],
        [(2.0, 2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
        [(-0.0, -0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)],
        [(float("nan"), 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)],
        [(0.0, 0.0, float("inf"), 1.0)],
        [(float("-inf"), 0.0, 0.0, 1.0), (0.0, 0.0, float("inf"), 1.0)],
        [(0.0, 0.0, 1e-160, 1e-160)],
        [(0.0, 0.0, 1e200, 1e200)],
    ],
)
def test_union_matches_frozen_python(rectangles: Sequence[tuple[float, float, float, float]]) -> None:
    assert_same_float(_rect_union_area(rectangles), reference_union(list(rectangles)))


def test_seeded_batches_match_python_including_order_and_duplicates() -> None:
    rng = random.Random(20261004)
    batches = []
    for _ in range(300):
        rectangles = []
        for _ in range(rng.randrange(21)):
            x, y, w, h = [rng.uniform(-1.0, 1.0) for _ in range(4)]
            rectangles.append((x, y, x + w, y + h))
        if len(rectangles) > 0:
            rectangles.append(rectangles[0])
        batches.append(rectangles)
    expected = [reference_union(rects) for rects in batches]
    for actual, reference in zip(_rect_union_areas(batches), expected, strict=True):
        assert_same_float(actual, reference)
    assert _rect_union_areas([]) == []


@pytest.mark.parametrize("lengths", [(1, 0, 1), (1, 1, 0), (0, 1, 1)])
def test_binding_rejects_misaligned_batches(lengths: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        rect_union_areas([[]] * lengths[0], [[]] * lengths[1], [[]] * lengths[2])


def test_page_and_group_scopes_match_frozen_python() -> None:
    rng = random.Random(28075)
    for _ in range(200):
        collections = []
        for _ in range(2):
            boxes = []
            for _ in range(rng.randrange(16)):
                x, y, w, h = (rng.uniform(-0.5, 1.0) for _ in range(4))
                boxes.append(
                    BBox(
                        page=rng.randrange(1, 4),
                        group=rng.choice([None, "a", "b"]),
                        bbox=(x, y, w, h),
                    )
                )
            collections.append(boxes)
        actual = compute_standard_iou_metrics(*collections)
        reference = reference_iou(*collections)
        for name in ("iou", "gt_area", "pred_area", "intersection_area", "union_area"):
            assert_same_float(getattr(actual, name), getattr(reference, name))


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), float("-inf"), -0.0])
def test_iou_nonfinite_coordinates_and_invalid_dimensions(coordinate: float) -> None:
    gt = [BBox(1, (coordinate, 0.0, 1.0, 1.0)), BBox(2, (0.0, 0.0, -1.0, 1.0))]
    pred = [BBox(1, (0.0, 0.0, 1.0, 1.0)), BBox(1, (0.0, 0.0, float("nan"), 1.0))]
    actual, reference = compute_standard_iou_metrics(gt, pred), reference_iou(gt, pred)
    for name in ("iou", "gt_area", "pred_area", "intersection_area", "union_area"):
        assert_same_float(getattr(actual, name), getattr(reference, name))


@pytest.mark.parametrize("overlap", [math.nextafter(0.5, 0.0), 0.5, math.nextafter(0.5, 1.0)])
def test_localization_threshold_uses_identical_iou(overlap: float) -> None:
    gt = [BBox(1, (0.0, 0.0, 1.0, 1.0))]
    pred = [BBox(1, (0.0, 0.0, overlap, 1.0))]
    actual, reference = compute_standard_iou_metrics(gt, pred), reference_iou(gt, pred)
    assert_same_float(actual.iou, reference.iou)
    assert field_grounding_localization_passes(iou=actual.iou, max_ioa=0.0, comparison=None) == (reference.iou >= 0.5)
