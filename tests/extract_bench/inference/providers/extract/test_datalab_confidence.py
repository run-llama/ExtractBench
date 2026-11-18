"""Fast extraction keeps its result when optional confidence polling fails."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("datalab_sdk")
pytest.importorskip("pypdf")

from aiohttp import web
from datalab_sdk import AsyncDatalabClient

from extract_bench.inference.cost_contract import (
    COMPLETE,
    COMPONENT_ID,
    COST_ACCOUNTING,
    COST_COMPONENT_EVENT,
    COST_PER_PAGE_USD,
    COST_USD,
    KNOWN_COST_USD,
)
from extract_bench.inference.limits import ExecutionBudget
from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.extract import datalab


@pytest.mark.parametrize(
    "outcome",
    [
        "scores",
        "ready",
        "turbo",
        "balanced",
        "accurate",
        "override",
        "no_cost",
        "delayed",
        "pending_deadline",
        "http_error",
        "invalid",
        "failed",
        "cancelled",
        "deadline",
        "signed",
        "invalid_url",
    ],
)
def test_extraction_survives_optional_scoring_with_separate_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
) -> None:
    async def run() -> None:
        calls = {"submit": 0, "poll": 0}
        clock = {"now": 0.0}
        monkeypatch.setattr(datalab, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))
        budget = ExecutionBudget(30)
        initial = {
            "success": True,
            "status": "complete",
            "page_count": 1,
            "cost_breakdown": None if outcome in ("turbo", "no_cost") else {"final_cost_cents": 1.7},
            "extraction_schema_json": '{"total":10}',
            "extraction_score_average": None,
            "extraction_scoring_status": "partial",
        }
        scored = {
            **initial,
            "extraction_schema_json": '{"total":10,"total_score":{"score":3}}',
            "extraction_score_average": 3,
        }

        async def submit(request: web.Request) -> web.Response:
            form = await request.post()
            assert form["extraction_mode"] == pipeline.config["extraction_mode"]
            assert form["citation_granularity"] == "phrase"
            if outcome == "override":
                assert form["mode"] == "fast"
            else:
                assert "mode" not in form
            calls["submit"] += 1
            return web.json_response({"success": True, "request_id": "job", "request_check_url": "/api/v1/extract/job"})

        async def poll(request: web.Request) -> web.Response:
            calls["poll"] += 1
            clock["now"] += 2.0 if calls["poll"] == 1 else 5.0
            if outcome == "ready":
                return web.json_response(scored)
            if calls["poll"] == 1 or (outcome == "delayed" and calls["poll"] < 66):
                return web.json_response(initial)
            if outcome == "pending_deadline":
                if calls["poll"] == 4:
                    budget.deadline = 0
                return web.json_response(initial)
            if outcome == "http_error":
                return web.json_response({"error": "unavailable"}, status=503)
            if outcome == "invalid":
                return web.json_response([])
            if outcome == "failed":
                return web.json_response(
                    {**initial, "extraction_schema_json": None, "extraction_scoring_status": "failed"},
                )
            if outcome == "cancelled":
                budget.cancel()
            if outcome == "deadline":
                budget.deadline = 0
            if outcome == "invalid_url":
                return web.json_response({"status": "complete", "result_url": {}})
            if outcome == "signed":
                return web.json_response({"status": "complete", "result_url": f"{base_url}/signed"})
            return web.json_response(scored)

        async def download(request: web.Request) -> web.Response:
            assert "X-API-Key" not in request.headers
            return web.json_response(scored)

        app = web.Application()
        app.router.add_post("/api/v1/extract", submit)
        app.router.add_get("/api/v1/extract/job", poll)
        app.router.add_get("/signed", download)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base_url = f"http://127.0.0.1:{runner.addresses[0][1]}"
        monkeypatch.setattr(
            datalab,
            "AsyncDatalabClient",
            lambda **kwargs: AsyncDatalabClient(base_url=base_url, **kwargs),
        )
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4 local test")
        extraction_mode = {
            "turbo": "turbo",
            "balanced": "balanced",
            "accurate": "accurate",
            "override": "balanced",
            "no_cost": "balanced",
        }.get(outcome, "fast")
        pipeline = get_pipeline(f"datalab_extract_{extraction_mode}")
        config = {**pipeline.config, "api_key": "local-test", "poll_interval": 0.001}
        if outcome == "override":
            config["parse_mode"] = "fast"
        provider = datalab.DatalabExtractProvider(
            "datalab_extract",
            config,
        )
        try:
            if outcome in ("invalid", "invalid_url"):
                with budget.activate(), pytest.raises(datalab.ProviderPermanentError) as caught:
                    await budget.run_async(provider._extract_pdf_async(str(source), {"type": "object"}))
                assert caught.value.debug_payload[COST_ACCOUNTING] == {KNOWN_COST_USD: 0.017, COMPLETE: True}
                return
            with budget.activate():
                result = await budget.run_async(provider._extract_pdf_async(str(source), {"type": "object"}))
            assert json.loads(result["extraction_schema_json"])["total"] == 10
            assert result["_config"]["citation_granularity"] == "phrase"
            assert calls["submit"] == 1
            assert result["extraction_latency_ms"] == 2000
            expected_polls = (
                1 if extraction_mode != "fast" else {"ready": 1, "delayed": 66, "pending_deadline": 4}.get(outcome, 2)
            )
            assert calls["poll"] == expected_polls
            if outcome == "no_cost":
                assert COST_USD not in result
            elif outcome == "turbo":
                assert result["parse_cost_usd"] == 0
                assert result[COST_USD] == pytest.approx(0.006)
            else:
                assert result[COST_USD] == pytest.approx(0.017)
                assert result[COST_PER_PAGE_USD] == pytest.approx(0.017)
            if outcome not in ("turbo", "override"):
                assert "datalab_pricing" not in result
            if extraction_mode != "fast":
                assert "confidence_wait_ms" not in result
                assert datalab._build_field_citations(json.loads(result["extraction_schema_json"]), {}) == []
                return
            assert result["confidence_wait_ms"] == (expected_polls - 1) * 5000
            if outcome in ("scores", "signed", "ready", "delayed"):
                assert result["_confidence_polling"]["status"] == "scores_ready"
                citations = datalab._build_field_citations(json.loads(result["extraction_schema_json"]), {})
                assert [(citation.field_path, citation.confidence, citation.bbox) for citation in citations] == [
                    ("total", 0.5, None)
                ]
            else:
                assert result["_confidence_polling"]["status"] in {
                    "poll_error",
                    "scoring_failed",
                    "deadline_exceeded",
                }
        finally:
            await runner.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("score", [None, True, "5", float("nan"), float("inf")])
def test_invalid_scores_do_not_become_confidence(score: object) -> None:
    assert datalab._build_field_citations({"total": 10, "total_score": {"score": score}}, {}) == []


def test_container_score_stays_on_parent_and_preserves_independent_child_score() -> None:
    citations = datalab._build_field_citations(
        {
            "items": [{"total": None, "total_score": {"score": 4}, "total_citations": ["unknown"], "unscored": 7}],
            "items_score": {"score": 5},
        },
        {},
    )
    assert [(citation.field_path, citation.confidence, citation.page) for citation in citations] == [
        ("items", 1.0, None),
        ("items[0].total", 0.75, None),
    ]


def test_missing_request_ids_do_not_merge_separate_extraction_charges(monkeypatch: pytest.MonkeyPatch) -> None:
    from datalab_sdk.models import ConversionResult

    from extract_bench.inference.events import bind_provider_event_sink

    class Client:
        base_url = "https://unused.invalid"

        async def __aenter__(self) -> "Client":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def extract(self, *args: object, **kwargs: object) -> ConversionResult:
            return ConversionResult(success=True, output_format="json", cost_breakdown={"final_cost_cents": 2})

    monkeypatch.setattr(datalab, "AsyncDatalabClient", lambda **kwargs: Client())
    monkeypatch.setattr(datalab, "_read_pdf_page_count", lambda path: 1)
    provider = datalab.DatalabExtractProvider("datalab_extract", {"api_key": "unused", "extraction_mode": "balanced"})
    charges = {}

    def record(event: str, payload: dict) -> None:
        if event == COST_COMPONENT_EVENT:
            charges[payload[COMPONENT_ID]] = payload[COST_USD]

    with bind_provider_event_sink(record):
        asyncio.run(provider._extract_pdf_async("unused.pdf", {}))
        asyncio.run(provider._extract_pdf_async("unused.pdf", {}))
    assert sum(charges.values()) == pytest.approx(0.04)


def _meta_sidecar() -> dict:
    return {
        "extraction_status": "EXTRACTED",
        "reasoning": None,
        "citations": ["/page/0/Table/7"],
        "verification": {"status": "PASS", "feedback": "ok"},
    }


def test_fast_scores_are_normalized_on_bracket_array_paths_without_citations() -> None:
    payload = {
        "line_items": [
            {
                "amount": 10.0,
                "amount_score": {"score": 3, "reasoning": "Moderate evidence"},
            }
        ]
    }

    citations = datalab._build_field_citations(payload, block_lookup={})

    assert len(citations) == 1
    assert citations[0].field_path == "line_items[0].amount"
    assert citations[0].confidence == 0.5
    assert citations[0].page is None
    assert citations[0].bbox is None
    assert citations[0].metadata == {
        "native_score": 3,
        "reasoning": "Moderate evidence",
        "confidence_semantics": "ordinal_1_5",
    }


@pytest.mark.parametrize(
    ("status", "expected"),
    [("PASS", 1.0), ("FAIL_FIX", 0.0), ("FAIL_CITATIONS", 0.0), ("ITEMS_MISSING", 0.0)],
)
def test_balanced_verification_is_a_binary_categorical_signal(status: str, expected: float) -> None:
    meta = _meta_sidecar()
    meta["verification"] = {"status": status, "feedback": "reviewed"}
    payload = {"invoice_number": "NF67652", "invoice_number_meta": meta}

    citations = datalab._build_field_citations(payload, block_lookup={})

    assert len(citations) == 1
    assert citations[0].confidence == expected
    assert citations[0].page is None
    assert citations[0].metadata["native_verification_status"] == status
    assert citations[0].metadata["confidence_semantics"] == "categorical_verification"
    assert citations[0].metadata["datalab_meta"]["verification"]["feedback"] == "reviewed"


def test_balanced_missing_verification_status_remains_unscored() -> None:
    payload = {"invoice_number": "NF67652", "invoice_number_meta": {"verification": {}}}

    assert datalab._build_field_citations(payload, block_lookup={}) == []


@pytest.mark.parametrize(("status", "expected"), [("PASS", None), ("FAIL_FIX", 0.0), ("FAIL_UNRESOLVABLE", 0.0)])
def test_explicit_verification_failure_is_retained_on_emitted_null(status: str, expected: float | None) -> None:
    payload = {
        "invoice_number": None,
        "invoice_number_citations": ["block-1"],
        "invoice_number_meta": {"verification": {"status": status}},
    }

    citations = datalab._build_field_citations(payload, block_lookup={})

    assert len(citations) == 1
    assert citations[0].confidence == expected
    assert citations[0].page is None


@pytest.mark.parametrize("status", ["PENDING", "NEW_STATUS"])
def test_inconclusive_verification_is_not_a_failed_score(status: str) -> None:
    citations = datalab._build_field_citations(
        {"amount": 10, "amount_citations": ["missing"], "amount_meta": {"verification": {"status": status}}},
        {},
    )
    assert len(citations) == 1
    assert citations[0].confidence is None
    assert citations[0].metadata["native_verification_status"] == status


@pytest.mark.parametrize("names", [[], ["Alice"]])
@pytest.mark.parametrize("mode", ["fast", "balanced"])
def test_array_confidence_reaches_the_value_f1_cell(names: list[str], mode: str) -> None:
    from extract_bench.evaluation.metrics.extract.confidence_scoped.scoring import build_confidence_field_rows
    from extract_bench.evaluation.metrics.extract.confidence_scoped.summary import confidence_signal_coverage_summary
    from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
        ValueCell,
        compute_unified_evidence_metrics,
    )

    data = {"names": names}
    payload: dict = {**data, "names_citations": ["unknown-block"]}
    if mode == "fast":
        payload["names_score"] = {"score": 5}
    else:
        payload["names_meta"] = {"verification": {"status": "PASS"}}
    citations = datalab._build_field_citations(payload, {})
    schema = {"properties": {"names": {"type": "array", "items": {"type": "string"}}}}
    cells: list[ValueCell] = []
    compute_unified_evidence_metrics(data, data, [], data_schema=schema, value_cells=cells)
    rows = build_confidence_field_rows(
        extracted_data=data,
        expected_output=data,
        field_rules=[],
        field_citations=citations,
        data_schema=schema,
        value_cells=cells,
    )

    assert [(row.predicted_path, row.confidence, row.correct) for row in rows] == [("names", 1.0, True)]
    assert confidence_signal_coverage_summary(data, citations, cells)["confidence_signal_coverage"] == 1.0


@pytest.mark.parametrize(("parent", "expected_score"), [(None, 0.0), ("PASS", 1.0), ("ITEMS_MISSING", 0.0)])
def test_scalar_item_verification_reaches_array_score(parent: str | None, expected_score: float) -> None:
    from extract_bench.evaluation.metrics.extract.confidence_scoped.scoring import build_confidence_field_rows
    from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
        ValueCell,
        compute_unified_evidence_metrics,
    )

    data = {"codes": ["A", "B", "C"]}
    schema = {"properties": {"codes": {"type": "array", "items": {"type": "string"}}}}
    meta = {
        "items": [
            {"verification": {"status": "PASS"}},
            {"verification": {"status": "FAIL_FIX"}, "citations": ["block-2"]},
            {"verification": {"status": "PENDING"}},
        ],
        "verification": {"status": parent},
    }
    citations = datalab._build_field_citations({**data, "codes_meta": meta}, {})
    items = {c.field_path: c for c in citations if c.field_path.startswith("codes[")}
    assert items["codes[0]"].confidence == 1.0
    assert items["codes[1]"].confidence == 0.0
    assert items["codes[1]"].metadata["block_id"] == "block-2"
    assert items["codes[1]"].metadata["native_verification_status"] == "FAIL_FIX"
    cells: list[ValueCell] = []
    compute_unified_evidence_metrics(data, data, [], data_schema=schema, value_cells=cells)
    rows = build_confidence_field_rows(
        extracted_data=data,
        expected_output=data,
        field_rules=[],
        field_citations=citations,
        data_schema=schema,
        value_cells=cells,
    )
    assert [(row.predicted_path, row.confidence) for row in rows] == [("codes", expected_score)]


def test_incomplete_positional_item_metadata_does_not_score_the_wrong_element() -> None:
    payload = {
        "codes": ["A", "B"],
        "codes_meta": {"items": [{"verification": {"status": "FAIL_FIX"}}]},
    }
    assert datalab._build_field_citations(payload, {}) == []
