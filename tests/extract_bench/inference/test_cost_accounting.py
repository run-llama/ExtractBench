"""Billing revisions, partial usage, and retries must not lose or duplicate charges."""

from typing import Any

import pytest

from extract_bench.inference.cost_accounting import (
    AttemptCost,
    preserve_cost_on_error,
    record_response_cost,
    summarize_attempts,
)
from extract_bench.inference.cost_contract import (
    COMPLETE,
    COMPONENT_ID,
    COMPONENTS,
    COST_ACCOUNTING,
    COST_ATTEMPT_EVENT,
    COST_COMPLETE,
    COST_COMPONENT_EVENT,
    COST_USD,
    KNOWN_COST_USD,
    OUTCOME,
    RAW_COST_COMPLETE,
    SUCCESSFUL_ATTEMPT_COST_USD,
    TOTAL_COST_USD,
    USAGE,
    VENDOR_USAGE_EVENT,
)
from extract_bench.inference.events import bind_provider_event_sink
from extract_bench.inference.providers.base import ProviderPermanentError


def test_revised_component_and_attempt_snapshots_replace_overlapping_estimates() -> None:
    cost = AttemptCost()
    cost.observe(COST_COMPONENT_EVENT, {COMPONENT_ID: "parse", COST_USD: 2})
    cost.observe(COST_COMPONENT_EVENT, {COMPONENT_ID: "extract", COST_USD: 3})
    cost.observe(COST_ATTEMPT_EVENT, {KNOWN_COST_USD: 5, COMPLETE: True})
    assert cost.known_cost_usd == 5
    assert cost.complete
    cost.observe(COST_COMPONENT_EVENT, {COMPONENT_ID: "extract", COST_USD: 1})
    assert cost.known_cost_usd == 5  # Earlier overlapping subtotal is provisional.
    assert not cost.complete
    cost.observe(COST_ATTEMPT_EVENT, {KNOWN_COST_USD: 3, COMPLETE: True})
    assert cost.known_cost_usd == 3
    assert cost.complete


def test_final_bill_below_known_charges_raises_without_replacing_prior_accounting() -> None:
    cost = AttemptCost()
    cost.observe(COST_COMPONENT_EVENT, {COMPONENT_ID: "extract", COST_USD: 0.12})
    with pytest.raises(ValueError, match="below recorded component charges"):
        cost.observe(COST_ATTEMPT_EVENT, {KNOWN_COST_USD: 0.10, COMPLETE: True})
    assert cost.known_cost_usd == 0.12
    assert not cost.complete


def test_float_roundoff_does_not_reject_matching_final_bill() -> None:
    cost = AttemptCost()
    for component_id, amount in [("parse", 0.1), ("extract", 0.2)]:
        cost.observe(COST_COMPONENT_EVENT, {COMPONENT_ID: component_id, COST_USD: amount})
    cost.observe(COST_ATTEMPT_EVENT, {KNOWN_COST_USD: 0.3, COMPLETE: True})
    assert cost.known_cost_usd == 0.3
    assert cost.complete


@pytest.mark.parametrize("usage_first", [True, False])
def test_usage_order_cannot_promote_partial_charge_to_complete(usage_first: bool) -> None:
    cost = AttemptCost()
    events = [
        (VENDOR_USAGE_EVENT, {COMPONENT_ID: "extract", USAGE: {"tokens": 42}}),
        (COST_COMPONENT_EVENT, {COMPONENT_ID: "extract", COST_USD: 1, COMPLETE: False}),
    ]
    for event, payload in events if usage_first else reversed(events):
        cost.observe(event, payload)
    cost.observe(COST_ATTEMPT_EVENT, {KNOWN_COST_USD: 1, COMPLETE: True})
    assert not cost.complete
    assert cost.snapshot()[COMPONENTS]["extract"][USAGE] == {"tokens": 42}
    cost.observe(COST_COMPONENT_EVENT, {COMPONENT_ID: "extract", COST_USD: 1, COMPLETE: True})
    cost.observe(COST_ATTEMPT_EVENT, {KNOWN_COST_USD: 1, COMPLETE: True})
    cost.observe(VENDOR_USAGE_EVENT, {COMPONENT_ID: "extract", USAGE: {"tokens": 42}})
    assert cost.complete


@pytest.mark.parametrize("amount,complete", [(None, False), (0, True)])
def test_unknown_bill_stays_distinct_from_explicit_zero(amount: float | None, complete: bool) -> None:
    cost = AttemptCost()
    cost.observe(COST_COMPONENT_EVENT, {COMPONENT_ID: "extract", COST_USD: amount})
    cost.observe(COST_ATTEMPT_EVENT, {KNOWN_COST_USD: amount, COMPLETE: complete})
    summary = summarize_attempts([{**cost.snapshot(), OUTCOME: "failure"}])
    assert summary[KNOWN_COST_USD] == amount
    assert summary[TOTAL_COST_USD] == amount
    assert summary[COST_COMPLETE] is complete


@pytest.mark.parametrize("settled", [True, False])
def test_timed_out_attempt_cost_is_included_alongside_successful_retry(settled: bool) -> None:
    cost = AttemptCost()
    cost.observe(COST_ATTEMPT_EVENT, {KNOWN_COST_USD: 2, COMPLETE: True})
    failed = {**cost.snapshot(), OUTCOME: "timeout", COST_COMPLETE: settled}
    summary = summarize_attempts([failed, {COST_USD: 3, COST_COMPLETE: True, OUTCOME: "success"}])
    assert summary[KNOWN_COST_USD] == 5
    assert summary[TOTAL_COST_USD] == (5 if settled else None)
    assert summary[SUCCESSFUL_ATTEMPT_COST_USD] == 3
    assert failed[OUTCOME] == "timeout"


def test_paid_response_keeps_charge_and_usage_when_validation_fails() -> None:
    cost = AttemptCost()
    usage: dict[str, Any] = {"tokens": 42}
    error = ProviderPermanentError("invalid output")
    with pytest.raises(ProviderPermanentError) as caught, bind_provider_event_sink(cost.observe):
        raw = {**record_response_cost(0.1, usage, component_id="response-1"), USAGE: usage}
        with preserve_cost_on_error(raw):
            raise error
    assert caught.value is error
    assert raw[RAW_COST_COMPLETE]
    assert error.debug_payload is not None
    cost.observe(COST_ATTEMPT_EVENT, error.debug_payload[COST_ACCOUNTING])
    assert cost.known_cost_usd == 0.1
    assert cost.complete
    assert cost.usage == usage
