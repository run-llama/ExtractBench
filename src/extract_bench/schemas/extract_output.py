"""Normalized output schema for extract tasks.

Re-exported from ``parse-bench`` so a harness pinning both packages can pass its
own provider output into this package's metrics: a second, structurally
identical ``ExtractOutput`` class would be rejected by Pydantic at every field
that declares one.
"""

from parse_bench.schemas.extract_output import ExtractOutput, FieldCitation

__all__ = ["ExtractOutput", "FieldCitation"]
