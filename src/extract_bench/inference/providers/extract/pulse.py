"""Provider for Pulse EXTRACT.

Pulse's current structured extraction flow is two-step:

1. POST /extract with the source file to create a saved extraction.
2. POST /schema with that extraction_id plus the benchmark JSON schema.

The older ``structured_output`` option on /extract is deprecated, so this
provider follows the documented Extract -> Schema path.
"""

import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

_API_BASE_URL = "https://api.runpulse.com"
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 900.0
_DEFAULT_JOB_TIMEOUT_SECONDS = 1800.0
_DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
_POLL_RETRYABLE_STATUS_CODES = {404, 429, 500, 502, 503, 504}
_JOB_SUCCESS_STATUSES = {"completed", "complete", "done"}
_JOB_FAILURE_STATUSES = {"failed", "error", "canceled", "cancelled", "expired"}
_RETRYABLE_SCHEMA_TERMINAL_ACTIONS = (
    "retry later",
    "retry the request",
    "please resubmit the request",
)


@dataclass
class _AttemptControl:
    """Cancellation token and absolute budget for one runner attempt."""

    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class _PulseAnchorGroup:
    """Resolved multi-anchor scalar without a user-schema key collision."""

    boxes: tuple[dict[str, Any], ...]


@register_provider("pulse_extract")
class PulseExtractProvider(Provider):
    """Provider for Pulse structured extraction via REST."""

    CREDIT_RATE_USD = 0.015

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        api_key = self.base_config.get("api_key") or os.getenv("PULSE_API_KEY")
        if not api_key or not isinstance(api_key, str):
            raise ProviderConfigError(
                "Pulse API key is required. Set PULSE_API_KEY environment variable or pass api_key in base_config."
            )
        self._api_key = api_key

        raw_base_url = (
            self.base_config.get("base_url")
            or self.base_config.get("api_base_url")
            or os.getenv("PULSE_API_BASE_URL")
            or _API_BASE_URL
        )
        if not isinstance(raw_base_url, str) or not raw_base_url.strip():
            raise ProviderConfigError("Pulse base URL must be a non-empty string")
        self._api_base_url = raw_base_url.strip().rstrip("/")
        if self._api_base_url.endswith("/extract"):
            self._api_base_url = self._api_base_url[: -len("/extract")]

        self._model: str | None = self.base_config.get("model")
        self._pages: str | None = self.base_config.get("pages")
        self._request_timeout = float(
            self.base_config.get(
                "request_timeout",
                self.base_config.get(
                    "timeout",
                    os.getenv("PULSE_REQUEST_TIMEOUT", _DEFAULT_REQUEST_TIMEOUT_SECONDS),
                ),
            )
        )
        self._job_timeout = float(
            self.base_config.get(
                "job_timeout",
                os.getenv("PULSE_JOB_TIMEOUT", _DEFAULT_JOB_TIMEOUT_SECONDS),
            )
        )
        self._poll_interval = float(self.base_config.get("poll_interval", os.getenv("PULSE_POLL_INTERVAL", 5.0)))
        self._poll_max_interval = float(self.base_config.get("poll_max_interval", 30.0))
        self._max_poll_errors = int(self.base_config.get("max_poll_errors", 12))
        self._run_timeout = float(self.base_config.get("run_timeout", _DEFAULT_RUN_TIMEOUT_SECONDS))
        self._async_run = bool(self.base_config.get("async_run", self.base_config.get("async", False)))
        self._schema_prompt: str | None = self.base_config.get("schema_prompt")
        self._schema_effort = bool(self.base_config.get("effort", self.base_config.get("schema_effort", False)))
        self._estimate_schema_cost = bool(self.base_config.get("estimate_schema_cost", True))

        for name, value in (
            ("request_timeout", self._request_timeout),
            ("job_timeout", self._job_timeout),
            ("poll_interval", self._poll_interval),
            ("poll_max_interval", self._poll_max_interval),
            ("run_timeout", self._run_timeout),
        ):
            if value <= 0:
                raise ProviderConfigError(f"{name} must be greater than zero")
        if self._max_poll_errors < 0:
            raise ProviderConfigError("max_poll_errors must be non-negative")

        self._inflight_lock = threading.Lock()
        self._attempts: dict[str, _AttemptControl] = {}
        self._inflight_jobs: dict[str, tuple[_AttemptControl, str]] = {}

        extensions = self.base_config.get("extensions")
        if extensions is not None and not isinstance(extensions, dict):
            raise ProviderConfigError("extensions must be a dict")
        self._extensions: dict[str, Any] | None = extensions

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key}

    def _handle_response(
        self,
        response: requests.Response,
        *,
        context: str,
        ambiguous_submission: bool = False,
    ) -> dict[str, Any]:
        if response.status_code in (401, 403):
            raise ProviderConfigError(
                f"Pulse auth failed during {context} ({response.status_code}): {response.text[:300]}"
            )
        if response.status_code == 429:
            raise ProviderRateLimitError(f"Pulse rate limit during {context} (429): {response.text[:300]}")
        if response.status_code in (500, 502, 503, 504) and not ambiguous_submission:
            raise ProviderTransientError(
                f"Pulse transient during {context} ({response.status_code}): {response.text[:300]}"
            )
        if response.status_code in (500, 502, 503, 504):
            raise ProviderPermanentError(
                f"Pulse {context} returned {response.status_code}; submission outcome is unknown, "
                "so it was not automatically retried",
                debug_payload={
                    "context": context,
                    "http_status": response.status_code,
                    "response": response.text[:2000],
                    "submission_outcome": "unknown",
                },
            )
        if 300 <= response.status_code < 400:
            raise ProviderPermanentError(
                f"Pulse returned an unexpected redirect during {context} ({response.status_code})"
            )
        if response.status_code >= 400:
            raise ProviderPermanentError(f"Pulse {context} failed ({response.status_code}): {response.text[:300]}")

        try:
            raw = response.json()
        except ValueError as e:
            raise ProviderPermanentError(f"Pulse returned non-JSON response during {context}: {e}") from e
        if not isinstance(raw, dict):
            raise ProviderPermanentError(f"Pulse returned unexpected {context} response type: {type(raw).__name__}")
        return raw

    def _retry_after_seconds(self, response: requests.Response, attempt: int) -> float:
        value: Any = response.headers.get("Retry-After")
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, Mapping):
            value = body.get("retry_after", value)
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            seconds = min(60.0, 2.0 * (1.7 ** min(attempt - 1, 7)))
        return max(0.1, min(90.0, seconds))

    def _remaining_seconds(
        self,
        control: _AttemptControl,
        *,
        stage_deadline: float | None = None,
    ) -> float:
        deadline = control.deadline
        if stage_deadline is not None:
            deadline = min(deadline, stage_deadline)
        return deadline - time.monotonic()

    def _ensure_active(
        self,
        control: _AttemptControl,
        *,
        context: str,
        stage_deadline: float | None = None,
        job_id: str | None = None,
    ) -> None:
        if control.cancelled.is_set():
            if job_id is not None:
                self._cancel_remote_job(job_id)
            raise ProviderPermanentError(
                f"Pulse {context} was cancelled by the benchmark runner",
                job_id=job_id,
                debug_payload={"context": context, "job_id": job_id, "reason": "runner_cancelled"},
            )

        now = time.monotonic()
        if now >= control.deadline:
            if job_id is not None:
                self._cancel_remote_job(job_id)
            raise ProviderPermanentError(
                f"Pulse {context} exceeded the shared {self._run_timeout:.0f}s document deadline",
                job_id=job_id,
                debug_payload={"context": context, "job_id": job_id, "reason": "document_deadline"},
            )
        if stage_deadline is not None and now >= stage_deadline:
            if job_id is not None:
                self._cancel_remote_job(job_id)
            raise ProviderPermanentError(
                f"Pulse {context} exceeded its stage deadline",
                job_id=job_id,
                debug_payload={"context": context, "job_id": job_id, "reason": "stage_deadline"},
            )

    def _bounded_request_timeout(
        self,
        control: _AttemptControl,
        *,
        context: str,
        stage_deadline: float | None = None,
    ) -> float:
        self._ensure_active(control, context=context, stage_deadline=stage_deadline)
        return max(
            0.001,
            min(
                self._request_timeout,
                self._remaining_seconds(control, stage_deadline=stage_deadline),
            ),
        )

    def _interruptible_wait(
        self,
        control: _AttemptControl,
        delay: float,
        *,
        context: str,
        stage_deadline: float | None = None,
        job_id: str | None = None,
    ) -> None:
        self._ensure_active(
            control,
            context=context,
            stage_deadline=stage_deadline,
            job_id=job_id,
        )
        remaining = self._remaining_seconds(control, stage_deadline=stage_deadline)
        control.cancelled.wait(min(max(0.0, delay), max(0.0, remaining)))
        self._ensure_active(
            control,
            context=context,
            stage_deadline=stage_deadline,
            job_id=job_id,
        )

    def _submit(
        self,
        request_fn: Callable[[float], requests.Response],
        *,
        context: str,
        control: _AttemptControl,
    ) -> requests.Response:
        """Submit once; a 429 is handed to the runner's retry ladder."""
        self._ensure_active(control, context=context)
        try:
            response = request_fn(self._bounded_request_timeout(control, context=context))
        except requests.RequestException as exc:
            raise ProviderPermanentError(
                f"Pulse {context} did not return a response; submission outcome is unknown, "
                "so it was not automatically retried",
                debug_payload={
                    "context": context,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "submission_outcome": "unknown",
                },
            ) from exc
        if response.status_code == 429:
            raise ProviderRateLimitError(
                f"Pulse {context} was rate-limited (429); no job was accepted",
                debug_payload={
                    "context": context,
                    "http_status": 429,
                    "response": response.text[:2000],
                },
            )
        return response

    def _same_origin(self, url: str) -> bool:
        base = urlparse(self._api_base_url)
        target = urlparse(url)
        return target.scheme == base.scheme and target.netloc == base.netloc

    def _fetch_large_result(
        self,
        raw: dict[str, Any],
        *,
        context: str,
        control: _AttemptControl,
    ) -> dict[str, Any]:
        if not raw.get("is_url") or not raw.get("url"):
            self._ensure_active(control, context=context)
            return raw

        url = urljoin(f"{self._api_base_url}/", str(raw["url"]))
        redirects = 0
        transient_attempt = 0
        result: dict[str, Any] = {}
        while True:
            self._ensure_active(control, context=f"{context} large-result download")
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ProviderPermanentError(f"Pulse {context} returned an invalid large-result URL")
            headers = self._headers() if self._same_origin(url) else {}
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=self._bounded_request_timeout(
                        control,
                        context=f"{context} large-result download",
                    ),
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                self._ensure_active(control, context=f"{context} large-result download")
                transient_attempt += 1
                if transient_attempt < 4:
                    self._interruptible_wait(
                        control,
                        min(8.0, 2.0 ** (transient_attempt - 1)),
                        context=f"{context} large-result retry",
                    )
                    continue
                raise ProviderPermanentError(
                    f"Pulse {context} large-result download failed after {transient_attempt} attempts: {exc}",
                    debug_payload={
                        "context": context,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "url_origin": urlparse(url).netloc,
                    },
                ) from exc

            self._ensure_active(control, context=f"{context} large-result download")
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location:
                    raise ProviderPermanentError(
                        f"Pulse {context} large-result redirect did not include a Location header"
                    )
                redirects += 1
                if redirects > 5:
                    raise ProviderPermanentError(f"Pulse {context} large-result download exceeded 5 redirects")
                url = urljoin(url, location)
                continue

            transient_attempt += 1
            if response.status_code in (429, 500, 502, 503, 504) and transient_attempt < 4:
                self._interruptible_wait(
                    control,
                    self._retry_after_seconds(response, transient_attempt),
                    context=f"{context} large-result retry",
                )
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                raise ProviderPermanentError(
                    f"Pulse {context} large-result download returned HTTP {response.status_code} after "
                    f"{transient_attempt} attempts; the accepted job was not resubmitted",
                    debug_payload={
                        "context": context,
                        "http_status": response.status_code,
                        "response": response.text[:2000],
                        "url_origin": urlparse(url).netloc,
                    },
                )
            result = self._handle_response(response, context=f"{context} large-result download")
            break

        nested = result.get("result")
        if isinstance(nested, dict) and len(result) == 1:
            result = nested
        for key in ("plan_info", "plan-info", "credits_used", "page_count"):
            if key in raw and key not in result:
                result[key] = raw[key]
        return result

    def _register_request(self, example_id: str) -> _AttemptControl:
        control = _AttemptControl(deadline=time.monotonic() + self._run_timeout)
        previous_job: str | None = None
        with self._inflight_lock:
            previous = self._attempts.get(example_id)
            if previous is not None:
                previous.cancelled.set()
            inflight = self._inflight_jobs.pop(example_id, None)
            if inflight is not None:
                inflight[0].cancelled.set()
                previous_job = inflight[1]
            self._attempts[example_id] = control
        if previous_job is not None:
            self._cancel_remote_job(previous_job)
        return control

    def _clear_request(self, example_id: str, control: _AttemptControl) -> None:
        with self._inflight_lock:
            if self._attempts.get(example_id) is control:
                self._attempts.pop(example_id, None)
            inflight = self._inflight_jobs.get(example_id)
            if inflight is not None and inflight[0] is control:
                self._inflight_jobs.pop(example_id, None)

    def _register_job(self, example_id: str, control: _AttemptControl, job_id: str) -> bool:
        with self._inflight_lock:
            if self._attempts.get(example_id) is not control or control.cancelled.is_set():
                return False
            self._inflight_jobs[example_id] = (control, job_id)
            return True

    def _clear_job(self, example_id: str, control: _AttemptControl, job_id: str) -> None:
        with self._inflight_lock:
            inflight = self._inflight_jobs.get(example_id)
            if inflight is not None and inflight[0] is control and inflight[1] == job_id:
                self._inflight_jobs.pop(example_id, None)

    def _cancel_remote_job(self, job_id: str) -> None:
        try:
            requests.delete(
                f"{self._api_base_url}/job/{job_id}",
                headers=self._headers(),
                timeout=min(self._request_timeout, 5.0),
                allow_redirects=False,
            )
        except requests.RequestException:
            pass

    def _poll_job(
        self,
        job_id: str,
        *,
        context: str,
        example_id: str,
        control: _AttemptControl,
    ) -> dict[str, Any]:
        """Poll one accepted job; poll failures never submit a replacement job."""
        self._ensure_active(control, context=context, job_id=job_id)
        if not self._register_job(example_id, control, job_id):
            self._cancel_remote_job(job_id)
            raise ProviderPermanentError(
                f"Pulse {context} job {job_id} belonged to a cancelled or superseded attempt",
                job_id=job_id,
            )
        started = time.monotonic()
        job_deadline = started + self._job_timeout
        started_at = datetime.now().isoformat()
        wait = self._poll_interval
        consecutive_errors = 0
        history: list[dict[str, Any]] = []
        last_status: str | None = None
        last_state: dict[str, Any] | None = None

        try:
            while True:
                elapsed = time.monotonic() - started
                self._ensure_active(
                    control,
                    context=f"{context} job {job_id}",
                    stage_deadline=job_deadline,
                    job_id=job_id,
                )
                if time.monotonic() >= job_deadline:
                    self._cancel_remote_job(job_id)
                    raise ProviderPermanentError(
                        f"Pulse {context} job {job_id} did not complete within {self._job_timeout:.0f}s; "
                        "the job was cancelled instead of submitting a duplicate",
                        job_id=job_id,
                        debug_payload={
                            "context": context,
                            "job_id": job_id,
                            "last_state": last_state,
                            "poll_history": history,
                        },
                    )

                self._interruptible_wait(
                    control,
                    min(wait, max(0.0, self._job_timeout - elapsed)),
                    context=f"{context} job {job_id}",
                    stage_deadline=job_deadline,
                    job_id=job_id,
                )
                try:
                    response = requests.get(
                        f"{self._api_base_url}/job/{job_id}",
                        headers=self._headers(),
                        timeout=self._bounded_request_timeout(
                            control,
                            context=f"{context} job {job_id} poll",
                            stage_deadline=job_deadline,
                        ),
                        allow_redirects=False,
                    )
                except requests.RequestException as exc:
                    self._ensure_active(
                        control,
                        context=f"{context} job {job_id} poll",
                        stage_deadline=job_deadline,
                        job_id=job_id,
                    )
                    consecutive_errors += 1
                    if consecutive_errors <= self._max_poll_errors:
                        wait = min(wait * 1.3, self._poll_max_interval)
                        continue
                    raise ProviderPermanentError(
                        f"Pulse {context} job {job_id} could not be polled after "
                        f"{consecutive_errors} consecutive transport errors; the accepted job was not resubmitted",
                        job_id=job_id,
                        debug_payload={
                            "context": context,
                            "job_id": job_id,
                            "exception_type": type(exc).__name__,
                            "exception": str(exc),
                            "last_state": last_state,
                            "poll_history": history,
                        },
                    ) from exc

                self._ensure_active(
                    control,
                    context=f"{context} job {job_id} poll",
                    stage_deadline=job_deadline,
                    job_id=job_id,
                )
                if response.status_code in _POLL_RETRYABLE_STATUS_CODES:
                    consecutive_errors += 1
                    if consecutive_errors <= self._max_poll_errors:
                        wait = min(wait * 1.3, self._poll_max_interval)
                        continue
                    raise ProviderPermanentError(
                        f"Pulse {context} job {job_id} polling returned HTTP {response.status_code} "
                        f"{consecutive_errors} times; the accepted job was not resubmitted",
                        job_id=job_id,
                        debug_payload={
                            "context": context,
                            "job_id": job_id,
                            "http_status": response.status_code,
                            "response": response.text[:2000],
                            "last_state": last_state,
                            "poll_history": history,
                        },
                    )

                state = self._handle_response(response, context=f"{context} poll")
                consecutive_errors = 0
                last_state = state
                status = str(state.get("status") or state.get("job_status") or "").lower()
                if status != last_status:
                    history.append(
                        {
                            "wall_clock": datetime.now().isoformat(),
                            "elapsed_s": round(time.monotonic() - started, 2),
                            "status": status or "unknown",
                            "created_at": state.get("created_at"),
                            "updated_at": state.get("updated_at"),
                        }
                    )
                    last_status = status

                if status in _JOB_SUCCESS_STATUSES:
                    result = state.get("result", state)
                    if not isinstance(result, dict):
                        raise ProviderPermanentError(
                            f"Pulse {context} job {job_id} completed without an object result",
                            job_id=job_id,
                            debug_payload={"context": context, "job_id": job_id, "state": state},
                        )
                    resolved = self._fetch_large_result(
                        result,
                        context=f"{context} job {job_id}",
                        control=control,
                    )
                    resolved["_pulse_job"] = {
                        "job_id": job_id,
                        "status": status,
                        "poll_started_at": started_at,
                        "poll_completed_at": datetime.now().isoformat(),
                        "total_elapsed_s": round(time.monotonic() - started, 2),
                        "poll_history": history,
                    }
                    return resolved
                if status in _JOB_FAILURE_STATUSES:
                    raise ProviderPermanentError(
                        f"Pulse {context} job {job_id} ended with status={status}: "
                        f"{state.get('error') or state.get('error_message') or 'no error detail'}",
                        job_id=job_id,
                        debug_payload={
                            "context": context,
                            "job_id": job_id,
                            "state": state,
                            "poll_history": history,
                        },
                    )

                wait = min(wait * 1.3, self._poll_max_interval)
        finally:
            self._clear_job(example_id, control, job_id)

    def _resolve_submission(
        self,
        raw: dict[str, Any],
        *,
        context: str,
        example_id: str,
        control: _AttemptControl,
    ) -> dict[str, Any]:
        job_id = raw.get("job_id")
        accepted_job_id = job_id if isinstance(job_id, str) and job_id else None
        self._ensure_active(control, context=context, job_id=accepted_job_id)
        status = str(raw.get("status") or "").lower()
        if isinstance(job_id, str) and job_id and (self._async_run or status in {"pending", "queued", "processing"}):
            if status in _JOB_SUCCESS_STATUSES and isinstance(raw.get("result"), dict):
                return self._fetch_large_result(raw["result"], context=context, control=control)
            return self._poll_job(
                job_id,
                context=context,
                example_id=example_id,
                control=control,
            )
        return self._fetch_large_result(raw, context=context, control=control)

    def _build_extract_fields(self) -> list[tuple[str, tuple[None, str]]]:
        fields: list[tuple[str, tuple[None, str]]] = []

        def add(name: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, bool):
                fields.append((name, (None, "true" if value else "false")))
            elif isinstance(value, (dict, list)):
                fields.append((name, (None, json.dumps(value))))
            else:
                fields.append((name, (None, str(value))))

        add("model", self._model)
        add("pages", self._pages)
        add("async", self._async_run or None)
        add("extensions", self._extensions)
        return fields

    def _extract_file(
        self,
        file_path: Path,
        *,
        example_id: str,
        control: _AttemptControl,
    ) -> dict[str, Any]:
        def submit(timeout: float) -> requests.Response:
            with file_path.open("rb") as f:
                files: list[tuple[str, Any]] = [("file", (file_path.name, f, _content_type_for_path(file_path)))]
                files.extend(self._build_extract_fields())
                return requests.post(
                    f"{self._api_base_url}/extract",
                    headers=self._headers(),
                    files=files,
                    timeout=timeout,
                    allow_redirects=False,
                )

        response = self._submit(
            submit,
            context="extract submission",
            control=control,
        )
        raw = self._handle_response(
            response,
            context="extract submission",
            ambiguous_submission=self._async_run,
        )
        return self._resolve_submission(
            raw,
            context="extract",
            example_id=example_id,
            control=control,
        )

    def _apply_schema(
        self,
        extraction_id: str,
        schema: dict[str, Any],
        *,
        example_id: str,
        control: _AttemptControl,
    ) -> dict[str, Any]:
        schema_config: dict[str, Any] = {
            "input_schema": schema,
            "effort": self._schema_effort,
        }
        if self._schema_prompt is not None:
            schema_config["schema_prompt"] = self._schema_prompt

        payload: dict[str, Any] = {
            "extraction_id": extraction_id,
            "schema_config": schema_config,
        }
        if self._async_run:
            payload["async"] = True

        def submit(timeout: float) -> requests.Response:
            return requests.post(
                f"{self._api_base_url}/schema",
                headers={**self._headers(), "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
                allow_redirects=False,
            )

        response = self._submit(
            submit,
            context="schema submission",
            control=control,
        )
        raw = self._handle_response(
            response,
            context="schema submission",
            ambiguous_submission=self._async_run,
        )
        return self._resolve_submission(
            raw,
            context="schema",
            example_id=example_id,
            control=control,
        )

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"PulseExtractProvider only supports EXTRACT product type, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override"
            )

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        started_at = datetime.now()
        control = self._register_request(request.example_id)
        try:
            extract_raw = self._extract_file(
                file_path,
                example_id=request.example_id,
                control=control,
            )
            self._ensure_active(control, context="between extract and schema")
            extraction_id = extract_raw.get("extraction_id")
            if not isinstance(extraction_id, str) or not extraction_id:
                raise ProviderPermanentError(f"Pulse /extract response did not include extraction_id: {extract_raw}")

            try:
                schema_raw = self._apply_schema(
                    extraction_id=extraction_id,
                    schema=request.schema_override,
                    example_id=request.example_id,
                    control=control,
                )
            except ProviderPermanentError as exc:
                if not _is_retryable_schema_terminal_error(exc):
                    raise
                raise ProviderTransientError(
                    str(exc),
                    job_id=exc.job_id,
                    debug_payload=exc.debug_payload,
                ) from exc
            self._ensure_active(control, context="schema completion")
        except (
            ProviderPermanentError,
            ProviderTransientError,
            ProviderConfigError,
            ProviderRateLimitError,
        ):
            raise
        except requests.Timeout as e:
            raise ProviderTransientError(f"Pulse request timed out: {e}") from e
        except requests.ConnectionError as e:
            raise ProviderTransientError(f"Pulse connection error: {e}") from e
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during Pulse extraction: {e}") from e
        finally:
            self._clear_request(request.example_id, control)

        raw_output: dict[str, Any] = {
            "extract": extract_raw,
            "schema": schema_raw,
            "_config": {
                "model": self._model,
                "pages": self._pages,
                "extensions": self._extensions,
                "schema_prompt": self._schema_prompt,
                "effort": self._schema_effort,
                "estimate_schema_cost": self._estimate_schema_cost,
                "async_run": self._async_run,
                "request_timeout": self._request_timeout,
                "job_timeout": self._job_timeout,
                "run_timeout": self._run_timeout,
                "poll_interval": self._poll_interval,
                "poll_max_interval": self._poll_max_interval,
            },
        }
        schema_job = _as_mapping(schema_raw.get("_pulse_job"))
        if isinstance(schema_job.get("job_id"), str):
            raw_output["job_id"] = schema_job["job_id"]
        _apply_usage_cost_fields(raw_output)

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=latency_ms,
        )

    def cancel(self, example_id: str) -> bool:
        """Cancel the accepted Pulse job for a runner watchdog timeout."""
        with self._inflight_lock:
            control = self._attempts.get(example_id)
            inflight = self._inflight_jobs.get(example_id)
            job_id = inflight[1] if inflight is not None else None
            if control is not None:
                control.cancelled.set()
        if control is None and job_id is None:
            return False
        if job_id is not None:
            self._cancel_remote_job(job_id)
        return True

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"PulseExtractProvider only supports EXTRACT product type, got {raw_result.product_type}"
            )

        _apply_usage_cost_fields(raw_result.raw_output)
        schema_output = _as_mapping(_as_mapping(raw_result.raw_output.get("schema")).get("schema_output"))
        values = schema_output.get("values", {})
        extracted_data = values if isinstance(values, (dict, list)) else {}

        # Pulse reports per-field citations as element-anchor ids (e.g. "tbl-1-r25c3",
        # "txt-2"), not geometry. The coordinates live separately in the /extract
        # payload's bounding_boxes table. Resolve each anchor id to the box Pulse
        # already reported for it, then reuse the shared citation collector. Boxes
        # are passed through unchanged (no rescaling or reprojection).
        anchor_index = _build_pulse_anchor_index(
            _as_mapping(raw_result.raw_output.get("extract")).get("bounding_boxes")
        )
        resolved_citations = _resolve_pulse_citation_anchors(schema_output.get("citations"), anchor_index)
        citations = _extract_pulse_field_citations(resolved_citations)

        output = ExtractOutput(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data if isinstance(extracted_data, (dict, list)) else {},
            field_citations=citations,
        )

        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )


def _content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        return f"image/{'jpeg' if suffix in {'.jpg', '.jpeg'} else suffix.lstrip('.')}"
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return "application/octet-stream"


def _is_retryable_schema_terminal_error(exc: ProviderPermanentError) -> bool:
    """A terminal schema failure that Pulse itself asks the caller to resubmit."""
    debug = _as_mapping(exc.debug_payload)
    state = _as_mapping(debug.get("state"))
    status = str(state.get("status") or state.get("job_status") or "").lower()
    if debug.get("context") != "schema" or status not in _JOB_FAILURE_STATUSES:
        return False
    error = str(state.get("error") or state.get("error_message") or "").lower()
    return any(action in error for action in _RETRYABLE_SCHEMA_TERMINAL_ACTIONS)


def _apply_usage_cost_fields(raw_output: dict[str, Any]) -> None:
    extract = _as_mapping(raw_output.get("extract"))
    schema = _as_mapping(raw_output.get("schema"))

    page_count = _coerce_float(extract.get("page_count"))
    plan_info = _as_mapping(extract.get("plan_info") or extract.get("plan-info"))
    pages_used = page_count
    if pages_used is None:
        pages_used = _coerce_float(plan_info.get("pages_used"))

    extract_credits = _coerce_float(extract.get("credits_used"))
    schema_credits = _coerce_float(schema.get("credits_used"))
    config = _as_mapping(raw_output.get("_config"))
    estimate_schema_cost = bool(config.get("estimate_schema_cost", True))
    extract_credits_estimated = False
    schema_credits_estimated = False
    if extract_credits is None and pages_used is not None:
        extract_credits = pages_used
        extract_credits_estimated = True
    if schema_credits is None and pages_used is not None and estimate_schema_cost and config.get("effort"):
        schema_credits = pages_used * 6
        schema_credits_estimated = True
    elif schema_credits is None and pages_used is not None and estimate_schema_cost:
        schema_credits = pages_used
        schema_credits_estimated = True

    if pages_used is not None:
        raw_output["num_pages"] = int(pages_used) if pages_used.is_integer() else pages_used
    if extract_credits is not None:
        raw_output["extract_credits_used"] = extract_credits
        raw_output["extract_cost_usd"] = extract_credits * PulseExtractProvider.CREDIT_RATE_USD
        raw_output["extract_credits_estimated"] = extract_credits_estimated
    if schema_credits is not None:
        raw_output["schema_credits_used"] = schema_credits
        raw_output["schema_cost_usd"] = schema_credits * PulseExtractProvider.CREDIT_RATE_USD
        raw_output["schema_credits_estimated"] = schema_credits_estimated

    total_credits = (extract_credits or 0.0) + (schema_credits or 0.0)
    if total_credits > 0:
        raw_output["credits_used"] = total_credits
        raw_output["cost_usd"] = total_credits * PulseExtractProvider.CREDIT_RATE_USD
        if pages_used and pages_used > 0:
            raw_output["cost_per_page_usd"] = raw_output["cost_usd"] / pages_used


def _extract_pulse_field_citations(citations: Any) -> list[FieldCitation]:
    return _dedupe(_collect_citations(citations, path=[]))


def _build_pulse_anchor_index(bounding_boxes: Any) -> dict[str, dict[str, Any]]:
    """Map Pulse element-anchor ids to the box Pulse reported for them.

    Pulse's citation payload references elements by id (``tbl-<n>-r<r>c<c>`` for
    table cells, ``txt-<n>`` for text blocks). The geometry for those ids lives in
    the /extract ``bounding_boxes`` payload, keyed the same way. Build a lookup so
    anchor citations can be grounded. Boxes are copied verbatim — the shared
    collector already understands the 8-value normalized-polygon shape, so no
    coordinate transform happens here.
    """
    index: dict[str, dict[str, Any]] = {}
    boxes = _as_mapping(bounding_boxes)

    for table in _as_sequence(boxes.get("Tables")):
        table = _as_mapping(table)
        table_id = _as_mapping(table.get("table_info")).get("id")
        if not isinstance(table_id, str) or not table_id:
            continue
        for cell in _as_sequence(table.get("cell_data")):
            cell = _as_mapping(cell)
            position = _as_mapping(cell.get("position"))
            row = _coerce_int(position.get("row"))
            column = _coerce_int(position.get("column"))
            if row is not None and column is not None:
                _register_pulse_anchor(
                    index,
                    cell,
                    fallback_anchor=f"{table_id}-r{row}c{column}",
                )

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            _register_pulse_anchor(index, node)
            for value in node.values():
                visit(value)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for value in node:
                visit(value)

    visit(bounding_boxes)
    return index


def _register_pulse_anchor(
    index: dict[str, dict[str, Any]],
    node: Any,
    *,
    fallback_anchor: str | None = None,
) -> None:
    node = _as_mapping(node)
    anchor = node.get("id") or fallback_anchor
    if not isinstance(anchor, str) or not anchor:
        return

    # Table cells expose location.coordinates; text blocks expose bounding_box.
    location = _as_mapping(node.get("location"))
    coordinates = (
        location.get("coordinates")
        if location
        else node.get("bounding_box") or node.get("bbox_normalized") or node.get("bbox") or node.get("coordinates")
    )
    if not isinstance(coordinates, Sequence) or isinstance(coordinates, (str, bytes, bytearray)):
        return

    resolved: dict[str, Any] = {"polygon": list(coordinates)}
    page = _coerce_int(location.get("page")) or _coerce_int(node.get("page")) or _coerce_int(node.get("page_number"))
    if page is not None:
        resolved["page"] = page
    for text_key in ("text", "content"):
        text = node.get(text_key)
        if isinstance(text, str) and text:
            resolved["text"] = text
            break
    confidence = _coerce_probability(node.get("confidence") or node.get("average_word_confidence"))
    if confidence is not None:
        resolved["confidence"] = confidence
    index[anchor] = resolved


def _resolve_pulse_citation_anchors(node: Any, index: dict[str, dict[str, Any]]) -> Any:
    """Replace each anchor-id leaf with its resolved box, preserving field paths.

    The citation tree mirrors the extracted-values tree, so keeping its dict/list
    shape lets the shared collector derive field paths like ``holdings[0].cusip``.
    Anchors that are empty or absent from the index resolve to ``None`` so no
    citation is emitted for that field.
    """
    if isinstance(node, str):
        exact = index.get(node.strip())
        if exact is not None:
            return dict(exact)

        anchor_boxes = [dict(index[anchor]) for part in node.split(",") if (anchor := part.strip()) and anchor in index]
        if not anchor_boxes:
            return None
        return _PulseAnchorGroup(tuple(anchor_boxes))
    if isinstance(node, Mapping):
        if _looks_like_citation(node):
            return dict(node)
        return {
            key: _resolve_pulse_citation_anchors(value, index) for key, value in node.items() if isinstance(key, str)
        }
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        return [_resolve_pulse_citation_anchors(item, index) for item in node]
    return None


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _collect_citations(node: Any, *, path: list[str]) -> list[FieldCitation]:
    if isinstance(node, _PulseAnchorGroup):
        field_path = _format_field_path(path)
        if not field_path:
            return []
        return [citation for item in node.boxes if (citation := _citation_from_node(field_path, item)) is not None]

    if isinstance(node, Mapping):
        field_path = _format_field_path(path)
        if field_path and _looks_like_citation(node):
            citation = _citation_from_node(field_path, node)
            return [citation] if citation is not None else []

        citations: list[FieldCitation] = []
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            citations.extend(_collect_citations(value, path=[*path, key]))
        return citations

    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        citations = []
        for index, item in enumerate(node):
            citations.extend(_collect_citations(item, path=[*path, f"[{index}]"]))
        return citations

    return []


def _looks_like_citation(node: Mapping[str, Any]) -> bool:
    return any(key in node for key in ("bbox", "bounding_box", "boundingBox", "polygon", "page", "page_number"))


def _citation_from_node(field_path: str, node: Mapping[str, Any]) -> FieldCitation | None:
    page = _coerce_int(node.get("page")) or _coerce_int(node.get("page_number"))
    if page is None:
        return None
    bbox, polygon = _extract_bbox_and_polygon(node)
    if bbox is None and not _has_page_only_citation(node):
        return None

    return FieldCitation(
        field_path=field_path,
        page=page,
        bbox=bbox,
        polygon=polygon,
        reference_text=_reference_text(node),
        confidence=_coerce_probability(node.get("confidence") or node.get("score")),
        source="pulse",
        metadata=_compact_metadata(node),
    )


def _extract_bbox_and_polygon(node: Mapping[str, Any]) -> tuple[list[float] | None, list[list[float]] | None]:
    raw = node.get("bbox", node.get("bounding_box", node.get("boundingBox", node.get("polygon"))))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None, None

    values = [_coerce_float(value) for value in raw]
    if any(value is None for value in values):
        return None, None
    coords = [float(value) for value in values if value is not None]

    if len(coords) == 8:
        points = [[coords[index], coords[index + 1]] for index in range(0, 8, 2)]
        if not _all_normalized(coords):
            return None, None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        left = min(xs)
        top = min(ys)
        bbox = [left, top, max(xs) - left, max(ys) - top]
        return _round_bbox(bbox), [[round(point[0], 8), round(point[1], 8)] for point in points]

    if len(coords) == 4:
        x1, y1, third, fourth = coords
        if not _all_normalized(coords):
            return None, None
        # Pulse schema examples use [x1, y1, x2, y2]. If the last two
        # coordinates cannot be a lower-right corner, fall back to xywh.
        if third > x1 and fourth > y1:
            return _round_bbox([x1, y1, third - x1, fourth - y1]), None
        return _round_bbox([x1, y1, third, fourth]), None

    return None, None


def _has_page_only_citation(node: Mapping[str, Any]) -> bool:
    return "page" in node or "page_number" in node


def _reference_text(node: Mapping[str, Any]) -> str | None:
    for key in ("text", "content", "value", "reference_text", "referenceText", "quote"):
        value = node.get(key)
        if isinstance(value, str):
            return value
    return None


def _compact_metadata(node: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = {
        key: value
        for key, value in node.items()
        if key not in {"bbox", "bounding_box", "boundingBox", "polygon", "page", "page_number"}
    }
    return dict(metadata) if metadata else None


def _format_field_path(path: list[str]) -> str:
    rendered = ""
    for token in path:
        if token.startswith("[") and token.endswith("]"):
            rendered += token
        elif rendered:
            rendered += "." + token
        else:
            rendered = token
    return rendered


def _dedupe(citations: list[FieldCitation]) -> list[FieldCitation]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[FieldCitation] = []
    for citation in citations:
        key = (
            citation.field_path,
            citation.page,
            tuple(citation.bbox) if citation.bbox is not None else None,
            citation.reference_text,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def _round_bbox(bbox: list[float]) -> list[float] | None:
    if not _valid_normalized_bbox(bbox):
        return None
    return [round(value, 8) for value in bbox]


def _valid_normalized_bbox(bbox: list[float]) -> bool:
    if len(bbox) != 4:
        return False
    x, y, width, height = bbox
    return (
        0 <= x <= 1
        and 0 <= y <= 1
        and 0 < width <= 1
        and 0 < height <= 1
        and x + width <= 1.000001
        and y + height <= 1.000001
    )


def _all_normalized(values: list[float]) -> bool:
    return all(0 <= value <= 1 for value in values)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_probability(value: Any) -> float | None:
    score = _coerce_float(value)
    if score is None or not 0.0 <= score <= 1.0:
        return None
    return score


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None
