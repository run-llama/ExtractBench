from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfidenceFieldRow:
    """Dashboard/evaluator row for one confidence-bearing emitted field."""

    field_path: str
    field_pattern: str
    predicted_path: str | None
    expected_value: Any
    predicted_value: Any
    confidence: float | None
    correct: bool
    comparison_score: float
    comparison_mode: str
    verified: bool
    alignment_mode: str = ""
    alignment_score: float = 0.0
    cited_text: str = ""
    schema_valid: bool = True
    matched_gt_path: str = ""
