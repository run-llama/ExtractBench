"""Extend Operator extraction on the 2026-02-09 API.

https://docs.extend.ai/extraction/operator
https://docs.extend.ai/general/operator-pricing
"""

import threading
from contextvars import ContextVar
from typing import Any

from extend_ai.core.api_error import ApiError

from extract_bench.inference.events import emit_provider_event
from extract_bench.inference.providers.base import ProviderPermanentError
from extract_bench.inference.providers.extract.extend import (
    USD_PER_CREDIT,
    ExtendProvider,
    _adapt_schema_for_extend,
    _coerce_nonnegative_float,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult


def _attach_operator_cost_stats(raw_output: dict[str, Any], pipeline_config: dict[str, Any]) -> None:
    """Use the measured roll-up, which already includes parsing and agent usage."""
    run = raw_output.get("processor_run") or {}
    files = run.get("files") or ([run["file"]] if run.get("file") else [])
    num_pages = sum(
        int((file.get("metadata") or {}).get("pageCount") or (file.get("metadata") or {}).get("page_count") or 0)
        for file in files
    )
    if num_pages:
        raw_output["num_pages"] = num_pages

    usage = run.get("usage") or {}
    total = _coerce_nonnegative_float(usage.get("totalCredits"))
    if total is None:
        total = _coerce_nonnegative_float(usage.get("total_credits"))
    if total is None:
        # Operator's usage.credits is also a total. Never add a parse estimate.
        total = _coerce_nonnegative_float(usage.get("credits"))
    if total is None:
        return

    rate = float(pipeline_config.get("usd_per_credit", USD_PER_CREDIT))
    raw_output.update(credits_used=total, total_credits=total, cost_usd=total * rate)
    if num_pages:
        raw_output["credits_per_page"] = total / num_pages
        raw_output["cost_per_page_usd"] = total * rate / num_pages

    charges = [charge for entry in usage.get("breakdown") or [] for charge in entry.get("charges") or []]
    # Keep every original charge in the run payload. Surface components only
    # when itemization exists, so missing agent usage is not reported as zero.
    if charges:
        raw_output["parse_credits"] = sum(
            charge["credits"] for charge in charges if charge.get("product", "").startswith("parse_")
        )
        raw_output["extract_credits"] = sum(
            charge["credits"] for charge in charges if charge.get("product") == "extraction_agent"
        )
        raw_output["agent_usage_credits"] = sum(
            charge["credits"] for charge in charges if charge.get("product") == "agent_usage"
        )
        raw_output["agent_usage_usd"] = sum(
            charge["quantity"]
            for charge in charges
            if charge.get("product") == "agent_usage" and charge.get("unit") == "usd"
        )


@register_provider("extend_operator")
class ExtendOperatorProvider(ExtendProvider):
    """Reuse Extend schema and citation normalization with modern extract runs."""

    _attach_cost_stats = staticmethod(_attach_operator_cost_stats)

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._active_runs: dict[str, threading.Event] = {}
        self._active_runs_lock = threading.Lock()
        # The shared Extend call path does not pass example IDs to _extract_document.
        # Context-local state binds its polling loop to the current worker's request.
        self._cancel_event: ContextVar[threading.Event] = ContextVar("extend_operator_cancel_event")

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        cancelled = threading.Event()
        with self._active_runs_lock:
            self._active_runs[request.example_id] = cancelled
        token = self._cancel_event.set(cancelled)
        try:
            return super().run_inference(pipeline, request)
        finally:
            self._cancel_event.reset(token)
            with self._active_runs_lock:
                if self._active_runs.get(request.example_id) is cancelled:
                    self._active_runs.pop(request.example_id)

    def cancel(self, example_id: str) -> bool:
        with self._active_runs_lock:
            cancelled = self._active_runs.get(example_id)
            if cancelled is None:
                return False
            cancelled.set()
        return True

    def _extract_document(
        self, file_path: str, schema: dict[str, Any], pipeline_config: dict[str, Any]
    ) -> dict[str, Any]:
        cancelled = self._cancel_event.get()
        run_id: str | None = None

        def check_cancelled() -> None:
            if cancelled.is_set():
                raise ProviderPermanentError(
                    f"Operator run {run_id} cancelled" if run_id is not None else "Operator extraction cancelled",
                    job_id=run_id,
                    debug_payload={"operator_run_id": run_id} if run_id is not None else None,
                )

        check_cancelled()
        adapted_schema, primitive_array_paths, renamed_props = _adapt_schema_for_extend(schema)

        def make_nullable(node: dict[str, Any]) -> None:
            # Unlike the legacy processor API, the current extract API rejects
            # non-nullable primitive fields and enums without a null option.
            # https://docs.extend.ai/extraction/schema#primitive-schema
            if node.get("type") in ("string", "number", "integer", "boolean"):
                node["type"] = [node["type"], "null"]
            if "enum" in node and None not in node["enum"]:
                node["enum"].append(None)
            for field in node.get("properties", {}).values():
                make_nullable(field)
            if isinstance(node.get("items"), dict):
                make_nullable(node["items"])

        make_nullable(adapted_schema)
        file_id = self._upload_file(file_path)
        check_cancelled()
        config = {
            "baseProcessor": "extraction_operator",
            "schema": adapted_schema,
            "advancedOptions": pipeline_config.get("advancedOptions", {}),
        }
        try:
            # Inline config avoids creating a persistent extractor per schema.
            # Disable SDK POST retries: a lost response may hide a billed run.
            response = self._client.extract_runs.create(
                config=config,  # type: ignore[arg-type]
                file={"id": file_id},
                request_options={"max_retries": 0},
            )
        except ApiError as error:
            if error.status_code == 429:
                self._handle_api_error(error, "Operator run create")
            raise ProviderPermanentError(f"Operator run creation failed: {error}") from error
        except Exception as error:
            raise ProviderPermanentError(
                f"Operator run creation response was lost. Check Extend before resubmitting: {error}"
            ) from error

        run = response.model_dump(mode="json", by_alias=True)
        run_id = run["id"]
        emit_provider_event(
            "vendor.job_created",
            provider="extend_operator",
            stage="extract",
            job_id=run_id,
        )
        consecutive_poll_errors = 0
        while run.get("status") in ("PENDING", "PROCESSING"):
            if cancelled.wait(self._poll_interval):
                check_cancelled()
            try:
                response = self._client.extract_runs.retrieve(run_id, request_options={"max_retries": 0})
                run = response.model_dump(mode="json", by_alias=True)
                consecutive_poll_errors = 0
            except Exception as error:
                consecutive_poll_errors += 1
                if (
                    isinstance(error, ApiError) and error.status_code not in (429, 500, 502, 503, 504)
                ) or consecutive_poll_errors > self._async_max_poll_errors:
                    # The harness must not create a new billed run after a
                    # polling failure. Include the existing ID for recovery.
                    raise ProviderPermanentError(f"Cannot poll Operator run {run_id}: {error}") from error

        if run.get("status") == "PROCESSED":
            emit_provider_event(
                "stage.completed",
                provider="extend_operator",
                stage="extract",
                job_id=run_id,
                raw_output=run,
            )
        check_cancelled()
        if run.get("status") != "PROCESSED":
            raise ProviderPermanentError(
                f"Operator run {run_id} ended with {run.get('status')}: "
                f"{run.get('failureReason')} {run.get('failureMessage')}"
            )

        # Keep the shared Extend envelope for schema/citation normalization.
        return {
            "processor_run": run,
            "_extend_metadata": {
                "file_id": file_id,
                "api_version": "2026-02-09",
                "primitive_array_paths": primitive_array_paths,
                "renamed_props": renamed_props,
            },
        }
