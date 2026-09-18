"""Rotated-bbox geometry helpers.

Re-exported from ``parse-bench`` so both packages agree on the rotated-box
representation (and on ``LiteralRotatedBox`` identity) when a harness pins both.
"""

from parse_bench.geometry.rotated_bbox import (
    LiteralRotatedBox,
    normalize_angle_degrees,
    polygon_angle_degrees,
    polygon_points,
    polygon_to_literal_xywh_r,
    rotated_rect_contains_point,
    xywh_r_to_polygon,
)

__all__ = [
    "LiteralRotatedBox",
    "normalize_angle_degrees",
    "polygon_angle_degrees",
    "polygon_points",
    "polygon_to_literal_xywh_r",
    "rotated_rect_contains_point",
    "xywh_r_to_polygon",
]
