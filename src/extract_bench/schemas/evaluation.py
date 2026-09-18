"""Schema definitions for evaluation results.

``MetricValue`` and ``RunStat`` are re-exported from ``parse-bench``. They are
what every metric returns, so a harness that pins both packages collects
metrics from both into one list: two structurally identical classes would make
that list unvalidatable. ``EvaluationResult`` and ``EvaluationSummary`` stay
local — this package carries diagnostic-metric fields that parse-bench has no
notion of, and a harness builds its own result model anyway.
"""

from datetime import datetime
from typing import Any

from parse_bench.schemas.evaluation import MetricValue, RunStat
from pydantic import BaseModel, Field

from extract_bench.schemas.metrics import ConfusionMatrixMetrics

__all__ = ["EvaluationResult", "EvaluationSummary", "MetricValue", "RunStat"]


class EvaluationResult(BaseModel):
    """Evaluation result for a single example."""

    test_id: str = Field(description="Test case identifier")
    example_id: str = Field(description="Example identifier from inference result")
    pipeline_name: str = Field(description="Pipeline that produced the result")
    product_type: str = Field(description="Product type (extract, parse, etc.)")
    success: bool = Field(description="Whether evaluation succeeded")
    metrics: list[MetricValue] = Field(default_factory=list, description="List of metric scores")
    diagnostic_metrics: list[MetricValue] = Field(
        default_factory=list,
        description="High-cardinality diagnostic metric scores excluded from headline metric menus",
    )
    error: str | None = Field(default=None, description="Error message if evaluation failed")
    evaluated_at: datetime = Field(default_factory=datetime.now, description="Timestamp when evaluation ran")
    job_id: str | None = Field(default=None, description="Provider job ID (e.g., LlamaExtract job UUID)")
    parse_job_id: str | None = Field(
        default=None,
        description="Parse job ID for the pipeline (LlamaParse job UUID)",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags from test case for filtering/grouping",
    )
    stats: list[RunStat] = Field(
        default_factory=list,
        description="Operational measurements (latency, cost, tokens, etc.)",
    )


class EvaluationSummary(BaseModel):
    """Aggregated evaluation metrics across all examples."""

    total_examples: int = Field(description="Total number of examples evaluated")
    successful: int = Field(description="Number of successful evaluations")
    failed: int = Field(description="Number of failed evaluations")
    skipped: int = Field(description="Number of skipped examples (no result found)")
    aggregate_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Aggregated metric values (e.g., avg_accuracy, avg_latency)",
    )
    aggregate_diagnostic_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Aggregated high-cardinality diagnostic metric values",
    )
    per_example_results: list[EvaluationResult] = Field(
        default_factory=list, description="Individual evaluation results"
    )
    confusion_matrix: ConfusionMatrixMetrics | None = Field(
        default=None,
        description=("Confusion matrix for layout detection evaluations (computed during evaluation)"),
    )
    started_at: datetime = Field(default_factory=datetime.now, description="When evaluation started")
    completed_at: datetime | None = Field(default=None, description="When evaluation completed")
    tag_metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description=("Per-tag aggregated metrics. Key=tag name, value=same format as aggregate_metrics"),
    )
    # Aggregate operational stats (latency, cost, tokens, etc.)
    aggregate_stats: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Aggregated stats keyed by stat name. "
            'Each value is {"total": ..., "avg": ..., "min": ..., "max": ..., '
            '"p50": ..., "p95": ..., "p99": ..., "count": ..., "unit": ...}'
        ),
    )
