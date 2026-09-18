"""Dotted field-path parsing and traversal for extract ground truth.

Re-exported from ``parse-bench``: field-path semantics must be identical in both
packages, since ground truth written against one is scored by the other.
"""

from parse_bench.test_cases.extract_field_paths import (
    get_path,
    inflate_expected_output,
    parse_field_path,
    set_path,
    validate_rules_match_expected_output,
)

__all__ = [
    "get_path",
    "inflate_expected_output",
    "parse_field_path",
    "set_path",
    "validate_rules_match_expected_output",
]
