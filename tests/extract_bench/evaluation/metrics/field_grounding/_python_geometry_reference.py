"""Frozen geometry oracle from ExtractBench c972087, independent of native output."""

from extract_bench.evaluation.metrics.field_grounding.core import (
    BBox,
    StandardIoUMetrics,
    _intersect_xyxy,
    _valid_xywh,
    _xywh_to_xyxy,
)


def compute_standard_iou_metrics(gt_boxes: list[BBox], pred_boxes: list[BBox]) -> StandardIoUMetrics:
    """Compute standard IoU between GT and predicted bbox sets.

    Rectangles are scoped by page and group. Within each scope, GT boxes and
    predicted boxes are independently unioned before intersection/union area
    are accumulated. This differs from :func:`compute_bbox_metrics`, whose
    historic ``iou`` field is GT-coverage shaped.
    """
    valid_gt = [box for box in gt_boxes if _valid_xywh(box.bbox)]
    valid_pred = [box for box in pred_boxes if _valid_xywh(box.bbox)]
    scopes = {(box.page, box.group) for box in valid_gt} | {(box.page, box.group) for box in valid_pred}

    gt_area = 0.0
    pred_area = 0.0
    intersection_area = 0.0
    for page, group in scopes:
        scope_gt = [box for box in valid_gt if box.page == page and box.group == group]
        scope_pred = [box for box in valid_pred if box.page == page and box.group == group]
        gt_rects = [_xywh_to_xyxy(box.bbox) for box in scope_gt]
        pred_rects = [_xywh_to_xyxy(box.bbox) for box in scope_pred]

        gt_area += _rect_union_area(gt_rects)
        pred_area += _rect_union_area(pred_rects)

        intersections: list[tuple[float, float, float, float]] = []
        for gt_rect in gt_rects:
            for pred_rect in pred_rects:
                if (intersection := _intersect_xyxy(gt_rect, pred_rect)) is not None:
                    intersections.append(intersection)
        intersection_area += _rect_union_area(intersections)

    union_area = gt_area + pred_area - intersection_area
    iou = intersection_area / union_area if union_area > 0.0 else 0.0
    return StandardIoUMetrics(
        iou=iou,
        gt_area=gt_area,
        pred_area=pred_area,
        intersection_area=intersection_area,
        union_area=union_area,
    )


def _rect_union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    if not rectangles:
        return 0.0

    xs = sorted({coord for rect in rectangles for coord in (rect[0], rect[2])})
    ys = sorted({coord for rect in rectangles for coord in (rect[1], rect[3])})
    total = 0.0
    for left, right in zip(xs, xs[1:], strict=False):
        if right <= left:
            continue
        for top, bottom in zip(ys, ys[1:], strict=False):
            if bottom <= top:
                continue
            if any(
                rect[0] <= left and rect[2] >= right and rect[1] <= top and rect[3] >= bottom for rect in rectangles
            ):
                total += (right - left) * (bottom - top)
    return total
