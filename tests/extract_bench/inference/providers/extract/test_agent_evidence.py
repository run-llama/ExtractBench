from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required")

from extract_bench.inference.pipelines.extract import register_extract_pipelines
from extract_bench.inference.providers.extract.agent_evidence import (
    agent_evidence_instruction,
    citations_from_agent_file,
    read_agent_citations_file,
)
from extract_bench.inference.providers.extract.claude_code_extract import ClaudeCodeExtractProvider
from extract_bench.inference.providers.extract.codex_code_extract import CodexCodeExtractProvider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType


def test_evidence_pipeline_configs_match_measured_agent_settings() -> None:
    specs: list[PipelineSpec] = []
    register_extract_pipelines(specs.append)
    by_name = {spec.pipeline_name: spec for spec in specs}

    claude = by_name["claude_code_extract_opus_4_8_evidence"]
    assert claude.provider_name == "claude_code_extract"
    assert claude.config == {"model": "claude-opus-4-8", "evidence_mode": True}

    for name, model in (
        ("codex_code_extract_gpt_5_5_low_evidence", "gpt-5.5"),
        ("codex_code_extract_gpt_5_6_sol_low_evidence", "gpt-5.6-sol"),
        ("codex_code_extract_gpt_5_6_terra_low_evidence", "gpt-5.6-terra"),
    ):
        evidence = by_name[name]
        assert evidence.config == {
            **by_name["codex_code_extract_gpt_5_5_low"].config,
            "model": model,
            "sandbox": "danger-full-access",
            "evidence_mode": True,
        }

    luna = by_name["codex_code_extract_gpt_5_6_luna_medium_evidence"]
    assert luna.config == {
        "model": "gpt-5.6-luna",
        "reasoning_effort": "medium",
        "sandbox": "danger-full-access",
        "max_cost_usd": None,
        "evidence_mode": True,
    }


@pytest.mark.parametrize("provider_class", [ClaudeCodeExtractProvider, CodexCodeExtractProvider])
def test_evidence_prompt_only_adds_the_shared_citation_rule(provider_class: type) -> None:
    base = provider_class("agent", {})
    evidence = provider_class("agent", {"evidence_mode": True})
    schema = {"type": "object", "properties": {"invoice_number": {"type": "string"}}}

    assert evidence._build_prompt(schema, "input.pdf") == (
        base._build_prompt(schema, "input.pdf") + "\n" + agent_evidence_instruction()
    )


def test_citations_match_agent_sidecar_rules(tmp_path: Path) -> None:
    citations_path = tmp_path / "citations.json"
    assert read_agent_citations_file(citations_path) is None
    citations_path.write_text("invalid json")
    assert read_agent_citations_file(citations_path) is None

    entries = [
        {"field_path": "line_items[0].amount", "page": 2, "bbox": [0.1, 0.2, 0.3, 0.4]},
        {"field_path": "invoice_number", "page": 1, "bbox": [0.9, 0.1, 0.3, 0.1]},
        {"field_path": "date", "page": 1, "bbox": "bad"},
        {"field_path": "total", "bbox": [0.1, 0.2, 0.1, 0.1]},
        {"page": 1},
    ]
    citations_path.write_text(json.dumps(entries))
    citations, stats = citations_from_agent_file(read_agent_citations_file(citations_path))

    assert [citation.field_path for citation in citations] == [
        "line_items[0].amount",
        "invoice_number",
        "date",
    ]
    assert citations[0].bbox == [0.1, 0.2, 0.3, 0.4]
    assert citations[1].bbox == [0.9, 0.1, 0.1, 0.1]
    assert citations[2].bbox is None
    assert stats.as_dict() == {
        "cells": 4,
        "with_page": 3,
        "with_bbox": 2,
        "malformed_bbox": 1,
        "unwrapped_cells": 0,
        "cells_with_extra_keys": 0,
        "dropped_entries": 1,
        "envelope_missing_data": 0,
    }


@pytest.mark.parametrize("provider_class", [ClaudeCodeExtractProvider, CodexCodeExtractProvider])
@pytest.mark.parametrize(
    ("citations_content", "expected_citations"),
    [
        ('[{"field_path": "invoice_number", "page": 1, "bbox": [0.1, 0.2, 0.3, 0.1]}]', 1),
        ("invalid json", 0),
        (None, 0),
    ],
)
def test_evidence_reaches_normalized_output(
    provider_class: type,
    citations_content: str | None,
    expected_citations: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "invoice.png"
    source.write_bytes(b"image")
    config = {"model": "claude-opus-4-8" if provider_class is ClaudeCodeExtractProvider else "gpt-5.5"}
    config["evidence_mode"] = True
    provider = provider_class("agent", config)
    pipeline = PipelineSpec(
        pipeline_name="agent_evidence",
        provider_name="agent",
        product_type=ProductType.EXTRACT,
        config=config,
    )
    request = InferenceRequest(
        example_id="invoice",
        source_file_path=str(source),
        product_type=ProductType.EXTRACT,
        schema_override={"type": "object", "properties": {"invoice_number": {"type": "string"}}},
    )

    def run_cli(*args: object) -> tuple[int, str]:
        workdir = next(arg for arg in args if isinstance(arg, Path) and arg.is_dir())
        (workdir / "output.json").write_text('{"invoice_number": "INV-1"}')
        if citations_content is not None:
            (workdir / "citations.json").write_text(citations_content)
        return 0, ""

    monkeypatch.setattr(provider, "_run_cli", run_cli)
    monkeypatch.setattr(provider, "_raise_for_status", lambda *args: None)
    if provider_class is ClaudeCodeExtractProvider:
        monkeypatch.setattr(provider, "_parse_result_event", lambda lines: {"usage": {}, "total_cost_usd": 0.01})
    else:
        monkeypatch.setattr(provider, "_raise_for_sandbox_failures", lambda lines: None)

    raw = provider.run_inference(pipeline, request)
    result = provider.normalize(raw)

    assert isinstance(raw.started_at, datetime)
    assert result.output.extracted_data == {"invoice_number": "INV-1"}
    assert len(result.output.field_citations) == expected_citations
    assert raw.raw_output["evidence_stats"]["with_bbox"] == expected_citations
    if expected_citations:
        assert result.output.field_citations[0].bbox == [0.1, 0.2, 0.3, 0.1]
