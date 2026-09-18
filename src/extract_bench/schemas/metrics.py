"""Schemas for evaluation metrics and confusion matrix data.

Re-exported from ``parse-bench``: these models cross the boundary between this
package's metrics and a downstream harness that also pins ``parse-bench``, so
both sides must see one class, not two structurally identical ones.
"""

from parse_bench.schemas.metrics import ConfusionMatrixCell, ConfusionMatrixMetrics

__all__ = ["ConfusionMatrixCell", "ConfusionMatrixMetrics"]
