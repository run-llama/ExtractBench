"""Pipeline specification.

Re-exported from ``parse-bench`` so pipeline specs registered by a downstream
harness are the same class this package's registries validate.
"""

from parse_bench.schemas.pipeline import PipelineSpec

__all__ = ["PipelineSpec"]
