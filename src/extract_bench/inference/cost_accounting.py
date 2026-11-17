"""Shared billing events and reconciliation for paid extraction attempts."""

import math
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from extract_bench.inference.cost_contract import (
    ATTEMPTS,
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
from extract_bench.inference.events import emit_provider_event
from extract_bench.inference.providers.base import ProviderError


def record_response_cost(
    cost_usd: float | None,
    usage: Mapping[str, Any],
    *,
    complete: bool = True,
    component_id: str | None = None,
) -> dict[str, Any]:
    """Publish one response before validation, including unknown or partial billing.

    A numeric amount may cover only reported counters. Carry its coverage with
    the component so a later error cannot promote a subtotal to a complete bill.
    """
    complete = cost_usd is not None and complete
    # Reusing an ID replaces its cumulative bill; separate calls in a composed
    # attempt must have separate IDs or one charge overwrites another. Prefer
    # the SDK response ID supplied by the caller. A local UUID covers SDKs
    # that omit an ID; this helper records each completed response only once.
    emit_provider_event(
        COST_COMPONENT_EVENT,
        **{
            COMPONENT_ID: component_id if component_id is not None else f"response:{uuid4().hex}",
            COST_USD: cost_usd,
            COMPLETE: complete,
            USAGE: dict(usage),
        },
    )
    return {COST_USD: cost_usd, RAW_COST_COMPLETE: complete}


@contextmanager
def preserve_cost_on_error(raw_output: Mapping[str, Any]) -> Generator[None]:
    """Attach the call's known bill without changing its failure or retry classification."""
    try:
        yield
    except ProviderError as error:
        error.debug_payload = {
            **(error.debug_payload or {}),
            USAGE: raw_output.get(USAGE),
            COST_ACCOUNTING: {
                KNOWN_COST_USD: raw_output.get(COST_USD),
                COMPLETE: raw_output.get(RAW_COST_COMPLETE, raw_output.get(COST_USD) is not None),
                USAGE: raw_output.get(USAGE),
            },
        }
        raise


SUCCESS_OUTCOME = "success"


def _validate_cost(value: Any) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
    ):
        raise ValueError("Cost must be a finite nonnegative number or null")


@dataclass
class AttemptCost:
    """Reconcile cumulative component charges with overlapping attempt snapshots.

    Reusing a component ID replaces its charge, including downward revisions;
    distinct IDs add. An attempt snapshot already includes those components.
    A revised component invalidates earlier full-attempt coverage until a new
    snapshot arrives. Its older subtotal remains provisional in the meantime.

    Callers serialize access and own persistence and execution outcomes. A final
    bill can be complete even when extraction failed or timed out.
    """

    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    usage: Any = None
    reported_cost_usd: float | None = None
    complete: bool = False

    @property
    def known_cost_usd(self) -> float | None:
        prices = [item[COST_USD] for item in self.components.values() if item[COST_USD] is not None]
        candidates = [self.reported_cost_usd] if self.reported_cost_usd is not None else []
        if len(prices) > 0:
            candidates.append(math.fsum(prices))
        if self.complete:
            return self.reported_cost_usd
        return max(candidates, default=None)

    def observe(self, event: str, payload: Mapping[str, Any]) -> None:
        """Consume billing events; unrelated provider telemetry does not change cost."""
        if event not in {COST_COMPONENT_EVENT, COST_ATTEMPT_EVENT, VENDOR_USAGE_EVENT}:
            return
        if event == COST_ATTEMPT_EVENT:
            value = payload[KNOWN_COST_USD]
            _validate_cost(value)
            complete = payload[COMPLETE]
            if not isinstance(complete, bool):
                raise ValueError("Attempt completeness must be boolean")
            prices = [item[COST_USD] for item in self.components.values() if item[COST_USD] is not None]
            component_total = math.fsum(prices)
            # Allow the measured summation error plus one representable step for
            # rounding the total (0.1 + 0.2 versus 0.3). One component involves no
            # summation, so its bill must match without any tolerance.
            roundoff = abs(sum(prices) - component_total) + math.ulp(component_total) if len(prices) > 1 else 0.0
            if complete and value is not None and component_total > value and component_total - value > roundoff:
                raise ValueError(f"Final attempt cost {value} is below recorded component charges {component_total}")
            if value is not None:
                self.reported_cost_usd = value
            self.complete = complete and value is not None and all(item[COMPLETE] for item in self.components.values())
            if USAGE in payload:
                self.usage = payload[USAGE]
            return

        component_id = payload[COMPONENT_ID]
        if not isinstance(component_id, str) or component_id == "":
            raise ValueError("Billing component must have a nonempty string ID")
        previous = self.components.get(component_id, {COST_USD: None, COMPLETE: False})
        updated = {
            **previous,
            **{key: value for key, value in payload.items() if key not in {COMPONENT_ID, COST_USD, COMPLETE}},
        }
        if event == COST_COMPONENT_EVENT:
            value = payload[COST_USD]
            _validate_cost(value)
            complete = payload.get(COMPLETE, value is not None)
            if not isinstance(complete, bool):
                raise ValueError("Component completeness must be boolean")
            updated[COST_USD] = value if value is not None else previous[COST_USD]
            updated[COMPLETE] = value is not None and complete
        # A usage-only notification preserves explicit billing coverage. Adding
        # an unpriced component or revising its bill requires a fresh snapshot.
        if component_id not in self.components or any(updated[key] != previous[key] for key in (COST_USD, COMPLETE)):
            self.complete = False
        self.components[component_id] = updated

    def snapshot(self) -> dict[str, Any]:
        return {
            COST_USD: self.known_cost_usd,
            COST_COMPLETE: self.complete,
            COMPONENTS: {key: dict(value) for key, value in self.components.items()},
            USAGE: self.usage,
        }


def summarize_attempts(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum distinct attempts, retaining incomplete charges and the latest success.

    Each input is a settled accounting snapshot with caller-owned identity and
    outcome fields. Callers mark unresolved execution incomplete before passing
    it here. A failed or timed-out attempt may still have a fully known bill.
    """
    known = [item[COST_USD] for item in attempts if item[COST_USD] is not None]
    subtotal = sum(known) if len(known) > 0 else None
    complete = len(attempts) > 0 and all(item[COST_COMPLETE] and item[COST_USD] is not None for item in attempts)
    successful = next((item for item in reversed(attempts) if item[OUTCOME] == SUCCESS_OUTCOME), None)
    return {
        ATTEMPTS: [dict(item) for item in attempts],
        KNOWN_COST_USD: subtotal,
        TOTAL_COST_USD: subtotal if complete else None,
        SUCCESSFUL_ATTEMPT_COST_USD: (
            successful[COST_USD] if successful is not None and successful[COST_COMPLETE] else None
        ),
        COST_COMPLETE: complete,
    }
