"""Base metric interface for evaluation metrics.

Re-exported from ``parse-bench`` so a metric written against either package's
``Metric`` is accepted by both, and ``isinstance`` checks hold across them.
"""

from parse_bench.evaluation.metrics.base import Metric

__all__ = ["Metric"]
