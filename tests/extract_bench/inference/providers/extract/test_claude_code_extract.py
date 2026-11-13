from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required")

from extract_bench.inference.providers.extract import claude_code_extract
from extract_bench.inference.providers.extract.claude_code_extract import ClaudeCodeExtractProvider


def _large_schema() -> dict[str, Any]:
    fields = {f"field_{i:04d}": {"type": "string", "description": "x" * 100} for i in range(1500)}
    return {"type": "object", "properties": fields}


def test_small_schema_stays_inline(tmp_path: Path) -> None:
    provider = ClaudeCodeExtractProvider("claude_code_extract")
    schema = provider._prepare_schema({"type": "object", "properties": {"invoice_number": {"type": "string"}}})
    prompt, delivery = provider._stage_prompt(schema, "input.pdf", tmp_path)
    assert delivery == "inline"
    assert prompt == provider._build_prompt(schema, "input.pdf")
    assert json.dumps(schema, indent=2) in prompt
    assert not (tmp_path / "schema.json").exists()


def test_oversized_schema_moves_to_file(tmp_path: Path) -> None:
    provider = ClaudeCodeExtractProvider("claude_code_extract", {"evidence_mode": True})
    schema = provider._prepare_schema(_large_schema())
    assert len(provider._build_prompt(schema, "input.pdf").encode()) > claude_code_extract._MAX_INLINE_PROMPT_BYTES

    prompt, delivery = provider._stage_prompt(schema, "input.pdf", tmp_path)
    assert delivery == "file"
    assert json.loads((tmp_path / "schema.json").read_text()) == schema
    assert "`./schema.json`" in prompt
    assert "field_0000" not in prompt
    assert len(prompt.encode()) < claude_code_extract._MAX_INLINE_PROMPT_BYTES
    # The rest of the ask, evidence rule included, is unchanged.
    assert prompt.endswith(provider._build_prompt(schema, "input.pdf").split("```\n\n", 1)[1])
