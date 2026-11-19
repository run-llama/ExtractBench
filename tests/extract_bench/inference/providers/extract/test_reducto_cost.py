"""The same Reducto usage can contain both current and legacy billing fields."""

import pytest

pytest.importorskip("reducto", reason="dev and runners extras required")
pytest.importorskip("pypdf", reason="dev and runners extras required")

from extract_bench.inference.cost_contract import COST_IS_ESTIMATE, COST_PER_PAGE_USD, COST_SOURCE, COST_USD
from extract_bench.inference.providers.extract.reducto import (
    ReductoCostSourceValues,
    attach_reducto_extract_cost,
)


@pytest.mark.parametrize("credits", [0, 8.0])
def test_returned_credits_override_both_page_counts(credits: float) -> None:
    raw = {
        "usage": {
            "credits": credits,
            "num_pages": 10,
            "num_fields": 3000,
            "usage_breakdown": {"extract_model": "Deep Extract", "extract_pages": 12},
        },
        "deep_extract_cost_calculated": True,
        "parse_credits": 10.0,
        COST_USD: 40.0,
        COST_IS_ESTIMATE: True,
    }
    attach_reducto_extract_cost(raw, {"deep_extract": True})
    assert raw[COST_USD] == pytest.approx(credits * 0.015)
    assert raw[COST_SOURCE] == ReductoCostSourceValues.REPORTED_CREDITS
    assert raw["credits_used"] == credits
    assert COST_IS_ESTIMATE not in raw
    assert "parse_credits" not in raw


@pytest.mark.parametrize(
    ("product", "tier", "pages", "mode", "expected", "source"),
    [
        ("Deep Extract", "Default", 12, "super_agent", 0.48, ReductoCostSourceValues.BILLED_PAGES),
        ("Deep Extract", "Default", 0, "super_agent", 0.0, ReductoCostSourceValues.BILLED_PAGES),
        ("Extract", "Batch", 8, "extract", 0.128, ReductoCostSourceValues.BILLED_PAGES),
        (None, None, None, "super_agent", 0.4, ReductoCostSourceValues.DOCUMENT_PAGES),
    ],
)
def test_page_estimate_uses_reported_billing_product_and_preferred_page_count(
    product: str | None,
    tier: str | None,
    pages: int | None,
    mode: str,
    expected: float,
    source: str,
) -> None:
    raw = {
        "usage": {
            "credits": None,
            "num_pages": 10,
            "num_fields": 3000,
            "extract_mode": mode,
            "usage_breakdown": {"extract_model": product, "tier": tier, "extract_pages": pages},
        },
    }
    attach_reducto_extract_cost(raw, {})
    first = raw.copy()
    attach_reducto_extract_cost(raw, {})
    assert raw == first
    assert raw[COST_USD] == pytest.approx(expected)
    assert raw[COST_PER_PAGE_USD] == pytest.approx(expected / 10)
    assert raw[COST_SOURCE] == source
    assert raw[COST_IS_ESTIMATE] is True
    assert "credits_used" not in raw


def test_unknown_billing_product_cannot_be_priced_from_the_request_defaults() -> None:
    raw = {"usage": {"num_pages": 10, "usage_breakdown": {"extract_model": "Future", "extract_pages": 10}}}
    attach_reducto_extract_cost(raw, {"deep_extract": True})
    assert COST_USD not in raw
