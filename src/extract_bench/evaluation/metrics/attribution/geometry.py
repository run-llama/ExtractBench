"""Re-export ParseBench layout attribution geometry."""

from parse_bench.evaluation.metrics.attribution.geometry import (
    coco_to_xyxy,
    compute_ioa,
    compute_ioa_matrix,
    compute_overlap_matrix,
    normalize_bbox_to_unit,
)

__all__ = [
    "coco_to_xyxy",
    "compute_ioa",
    "compute_ioa_matrix",
    "compute_overlap_matrix",
    "normalize_bbox_to_unit",
]
