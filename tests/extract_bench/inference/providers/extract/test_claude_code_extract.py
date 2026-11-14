from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required")

from extract_bench.inference.cost_contract import (
    COMPLETE,
    COST_ACCOUNTING,
    COST_COMPONENT_EVENT,
    COST_USD,
    KNOWN_COST_USD,
    USAGE,
)
from extract_bench.inference.events import bind_provider_event_sink
from extract_bench.inference.providers.base import ProviderPermanentError
from extract_bench.inference.providers.extract import claude_code_extract
from extract_bench.inference.providers.extract.claude_code_extract import ClaudeCodeExtractProvider
from extract_bench.inference.providers.extract.cli_harness import CliHarness
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType


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


@pytest.mark.parametrize("terminal_error", [False, True])
def test_claude_retains_known_terminal_cost_when_output_cannot_be_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, terminal_error: bool
) -> None:
    def fake_run(
        self: ClaudeCodeExtractProvider, cmd: list[str], workdir: Path, example_id: str, lines: list[str]
    ) -> tuple[int, str]:
        lines.append(
            json.dumps(
                {
                    "type": "result",
                    "subtype": "error_during_execution" if terminal_error else "success",
                    "is_error": terminal_error,
                    "total_cost_usd": 1.25,
                    "usage": {"input_tokens": 1_000},
                }
            )
        )
        return int(terminal_error), ""

    monkeypatch.setattr(ClaudeCodeExtractProvider, "_run_cli", fake_run)
    monkeypatch.setattr(CliHarness, "prepare", lambda *args: {"executable": "claude"})
    monkeypatch.setattr(claude_code_extract, "page_count", lambda _: 1)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"test input")
    pipeline = PipelineSpec(pipeline_name="test", provider_name="claude_code_extract", product_type=ProductType.EXTRACT)
    request = InferenceRequest(
        example_id="test",
        source_file_path=str(source),
        product_type=ProductType.EXTRACT,
        schema_override={"type": "object", "properties": {"invoice_number": {"type": "string"}}},
    )
    events = []
    with (
        bind_provider_event_sink(lambda name, payload: events.append((name, payload))),
        pytest.raises(ProviderPermanentError, match="error_during_execution|output.json") as excinfo,
    ):
        ClaudeCodeExtractProvider("claude_code_extract").run_inference(pipeline, request)

    expected = {KNOWN_COST_USD: 1.25, COMPLETE: True, USAGE: {"input_tokens": 1_000}}
    ((name, component),) = events
    assert name == COST_COMPONENT_EVENT
    assert component[COST_USD] == expected[KNOWN_COST_USD]
    assert component[USAGE] == expected[USAGE]
    assert excinfo.value.debug_payload[COST_ACCOUNTING] == expected
