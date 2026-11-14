"""Agentic extraction provider that shells out to the pi coding agent CLI (`pi -p`).

Same harness shape as ``claude_code_extract`` / ``codex_code_extract``: each
document is copied into an isolated temporary workdir, the agent is asked to
inspect that local file with shell tools and write ``./output.json``, and the
result is normalized through the standard Extract output path.

pi's provider registration is pure JSON config (``models.json``) and it speaks
several wire protocols natively (``openai-completions`` here), so this provider
talks to any OpenAI-compatible endpoint with no proxy. A fresh ``models.json``
is written into a per-request pi home directory (``PI_CODING_AGENT_DIR``),
beside the workdir rather than inside it, so concurrent requests never share
provider config, the agent never sees the API key file, and the user's real
``~/.pi`` is never touched.

pi reports its own dollar cost (``message.usage.cost.total``) computed from the
per-million rates in ``models.json``, so each pipeline puts real pricing into
the generated model entry and this provider reads the cost back rather than
re-deriving it.

Each document gets one CLI run with no time limit of its own; the runner's
per-file deadline cancels it through ``cancel``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from extract_bench.inference.cost_accounting import preserve_cost_on_error, record_response_cost
from extract_bench.inference.cost_contract import (
    COST_PER_PAGE_USD,
    COST_USD,
    USAGE,
)
from extract_bench.inference.limits import execution_budget
from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.agent_evidence import (
    agent_evidence_instruction,
    citations_from_agent_file,
    read_agent_citations_file,
)
from extract_bench.inference.providers.extract.cli_harness import HARNESS_VERSION, CliHarness
from extract_bench.inference.providers.extract.direct_model_utils import (
    add_additional_properties_false,
    force_all_properties_required,
    normalize_extract_result,
    page_count,
    promote_repeated_structure,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

_TRACE_TAIL_EVENTS = 20

_RATE_LIMIT_KEYWORDS = ("rate limit", "rate_limit", "429", "quota")
_TRANSIENT_KEYWORDS = (
    "overloaded",
    "high demand",
    "529",
    "503",
    "502",
    "504",
    "500",
    "408",
    "timeout",
    "timed out",
    "econnreset",
    "econnrefused",
    "socket hang up",
)

# pi gates model availability on "auth configured", so an endpoint without auth
# still needs a non-empty key.
_API_KEY_PLACEHOLDER = "unused"
_PRICE_KEYS = ("input_price_per_1m", "output_price_per_1m", "cache_read_price_per_1m")


@dataclass
class PiEventSummary:
    """Everything downstream code needs from one pi NDJSON trace, in a single pass."""

    usage: dict[str, int]
    cost: dict[str, float]
    cost_observed: bool
    cost_complete: bool
    error_messages: list[str]
    tool_diagnostics: dict[str, Any]
    trace_tail: list[Any]


def pi_reported_cost(usage: Mapping[str, Any]) -> float | None:
    cost = usage.get("cost")
    total = cost.get("total") if isinstance(cost, dict) else None
    if isinstance(total, (int, float)) and not isinstance(total, bool) and math.isfinite(total) and total >= 0:
        return float(total)
    return None


def summarize_pi_events(lines: list[str]) -> PiEventSummary:
    """Parse pi's NDJSON trace once and derive usage, cost, errors, and tool diagnostics.

    Each assistant ``message_end`` carries the usage for that one completion
    call (not a running session total), so summing every turn gives the run
    total. pi also computes a dollar cost per turn from the ``models.json``
    rates; sum that too so ``usage.cost.total`` can be read back directly.

    Error detection reads ``errorMessage`` off any ``message_end`` whose
    ``stopReason`` is ``"error"``, independent of exit code: pi exits 0 even
    when a turn fails at the model/transport level, so failure detection
    cannot rely on the process return code alone.
    """
    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
    }
    cost = {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, "total": 0.0}
    cost_observed = False
    cost_complete = True
    error_messages: list[str] = []
    tool_completed = tool_succeeded = tool_failed = 0
    tool_failures: list[dict[str, Any]] = []

    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")

        if event_type == "message_end":
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            if message.get("stopReason") == "error":
                error_message = message.get("errorMessage")
                if isinstance(error_message, str) and error_message:
                    error_messages.append(error_message)
            if message.get("role") != "assistant":
                continue
            turn_usage = message.get(USAGE)
            if not isinstance(turn_usage, dict):
                cost_complete = False
                continue
            usage["input_tokens"] += int(turn_usage.get("input") or 0)
            usage["cached_input_tokens"] += int(turn_usage.get("cacheRead") or 0)
            usage["cache_write_tokens"] += int(turn_usage.get("cacheWrite") or 0)
            usage["output_tokens"] += int(turn_usage.get("output") or 0)
            usage["reasoning_output_tokens"] += int(turn_usage.get("reasoning") or 0)
            usage["total_tokens"] += int(turn_usage.get("totalTokens") or 0)
            turn_cost = turn_usage.get("cost")
            turn_total = pi_reported_cost(turn_usage)
            if turn_total is not None and isinstance(turn_cost, dict):
                cost_observed = True
                cost["input"] += float(turn_cost.get("input") or 0)
                cost["output"] += float(turn_cost.get("output") or 0)
                cost["cache_read"] += float(turn_cost.get("cacheRead") or 0)
                cost["cache_write"] += float(turn_cost.get("cacheWrite") or 0)
                cost["total"] += turn_total
            else:
                cost_complete = False

        elif event_type == "tool_execution_end":
            tool_completed += 1
            if event.get("isError"):
                tool_failed += 1
                tool_failures.append(
                    {"tool": event.get("toolName"), "args": event.get("args"), "result": event.get("result")}
                )
            else:
                tool_succeeded += 1

    trace_tail: list[Any] = []
    for line in lines[-_TRACE_TAIL_EVENTS:]:
        try:
            trace_tail.append(json.loads(line))
        except json.JSONDecodeError:
            trace_tail.append(line)

    tool_diagnostics = {
        "completed": tool_completed,
        "succeeded": tool_succeeded,
        "failed": tool_failed,
        "failures": tool_failures[:5],
    }
    return PiEventSummary(
        usage=usage,
        cost=cost,
        cost_observed=cost_observed,
        cost_complete=cost_observed and cost_complete,
        error_messages=error_messages,
        tool_diagnostics=tool_diagnostics,
        trace_tail=trace_tail,
    )


@register_provider("pi_extract")
class PiExtractProvider(Provider):
    """Agentic extraction by shelling out to the pi coding agent CLI (`pi -p`)."""

    DEFAULT_WIRE_API = "openai-completions"
    DEFAULT_PROVIDER_LABEL = "openai-compatible"
    DEFAULT_MAX_COST_USD = 5.00
    DEFAULT_MAX_TOKENS = 32_000
    DEFAULT_CONTEXT_WINDOW = 1_000_000

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        missing = [key for key in ("model", "base_url", *_PRICE_KEYS) if key not in self.base_config]
        if missing:
            raise ProviderConfigError(f"pi_extract requires pipeline config keys: {', '.join(missing)}")
        self._model: str = self.base_config["model"]
        self._base_url: str = self.base_config["base_url"]
        self._wire_api: str = self.base_config.get("wire_api", self.DEFAULT_WIRE_API)
        self._provider_label: str = self.base_config.get("provider_label", self.DEFAULT_PROVIDER_LABEL)
        # Key resolution: explicit config value, then the env var the pipeline
        # names, then the non-empty placeholder pi requires.
        self._api_key_env: str = str(self.base_config.get("api_key_env") or "PI_EXTRACT_API_KEY")
        self._api_key: str = self.base_config.get("api_key") or os.getenv(self._api_key_env) or _API_KEY_PLACEHOLDER
        self._max_cost_usd: float | None = self.base_config.get("max_cost_usd", self.DEFAULT_MAX_COST_USD)
        self._max_tokens: int = int(self.base_config.get("max_tokens", self.DEFAULT_MAX_TOKENS))
        self._context_window: int = int(self.base_config.get("context_window", self.DEFAULT_CONTEXT_WINDOW))
        self._pi_bin: str = self.base_config.get("pi_bin", "pi")
        self._harness = CliHarness("pi_extract", self._pi_bin, self.base_config.get(HARNESS_VERSION))

        self._promote_repeated: bool = bool(self.base_config.get("promote_repeated_structure", True))
        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))
        self._all_properties_required: bool = bool(self.base_config.get("all_properties_required", False))
        # Evidence mode: ask the agent to also write ./citations.json with a page
        # + bbox per extracted value. Same side-file contract as claude_code_extract
        # / codex_code_extract.
        self._evidence_mode: bool = bool(self.base_config.get("evidence_mode", False))
        # Extra model-specific guidance appended to the prompt.
        self._extra_instructions: str = str(self.base_config.get("extra_instructions") or "").strip()
        # Request-level sampling overrides, emitted as the model entry's
        # ``samplingParams`` in models.json. pi Object.assigns these onto every
        # completion request, e.g. ``reasoning_effort``.
        raw_sampling = self.base_config.get("sampling_params")
        self._sampling_params: dict[str, Any] = dict(raw_sampling) if isinstance(raw_sampling, dict) else {}
        # Vision: declare image input in models.json. pi's read tool then sends
        # rendered page PNGs as image attachments.
        self._enable_vision: bool = bool(self.base_config.get("enable_vision", False))

        self._input_price_per_1m: float = float(self.base_config["input_price_per_1m"])
        self._output_price_per_1m: float = float(self.base_config["output_price_per_1m"])
        self._cache_read_price_per_1m: float = float(self.base_config["cache_read_price_per_1m"])

        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._procs_lock = threading.Lock()

    def _prepare_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if self._promote_repeated:
            schema = promote_repeated_structure(schema)
        if self._additional_properties_false:
            schema = add_additional_properties_false(schema)
        if self._all_properties_required:
            schema = force_all_properties_required(schema)
        return schema

    def _build_prompt(self, schema: dict[str, Any], staged_name: str) -> str:
        schema_json = json.dumps(schema, indent=2)
        evidence_rule = f"\n{agent_evidence_instruction()}" if self._evidence_mode else ""
        return (
            f"Extract structured data from the document file `./{staged_name}` in the current directory.\n\n"
            "Use only local file inspection and shell commands. Do not use web search, network calls, "
            "browser tools, or external services. Temporary scratch files inside the current directory are OK.\n\n"
            "Return a single JSON object conforming to this schema as your final answer:\n\n"
            f"```json\n{schema_json}\n```\n\n"
            "Rules:\n"
            "- Use null for fields not present in the document.\n"
            "- For list/array fields, enumerate every relevant row visible in the document; never collapse rows.\n"
            "- For large regular tables, prefer writing and running a local script to parse/enumerate rows.\n"
            "- For forms, prefer direct field extraction from the document content.\n"
            "- Write the resulting JSON object to ./output.json and validate that it is valid JSON before stopping.\n"
            f"- Do not print the JSON to your assistant output.{evidence_rule}"
            + (f"\n\n{self._extra_instructions}" if self._extra_instructions else "")
        )

    def _build_models_json(self) -> dict[str, Any]:
        """Per-request pi provider config, written to a per-request PI_CODING_AGENT_DIR."""
        model_entry: dict[str, Any] = {
            "id": self._model,
            "name": f"{self._model} (via pi_extract)",
            "reasoning": False,
            "input": ["text", "image"] if self._enable_vision else ["text"],
            "contextWindow": self._context_window,
            "maxTokens": self._max_tokens,
            "cost": {
                "input": self._input_price_per_1m,
                "output": self._output_price_per_1m,
                "cacheRead": self._cache_read_price_per_1m,
                "cacheWrite": 0,
            },
        }
        if self._sampling_params:
            model_entry["samplingParams"] = self._sampling_params
        return {
            "providers": {
                self._provider_label: {
                    "baseUrl": self._base_url,
                    "api": self._wire_api,
                    "apiKey": self._api_key or _API_KEY_PLACEHOLDER,
                    "models": [model_entry],
                }
            }
        }

    def _build_cmd(self) -> list[str]:
        # The prompt goes over stdin, not argv: it embeds the whole JSON schema,
        # and a large one pushes the command line past ARG_MAX. `-p` with no
        # positional message makes pi read stdin.
        # The --no-* flags keep the run hermetic: pi otherwise discovers skills,
        # extensions, prompt templates and AGENTS.md from the host's home directory.
        return [
            self._pi_bin,
            "--provider",
            self._provider_label,
            "--model",
            self._model,
            "--mode",
            "json",
            "--approve",
            "--no-session",
            "--no-skills",
            "--no-extensions",
            "--no-prompt-templates",
            "--no-context-files",
            "-p",
        ]

    def _run_cli(
        self,
        cmd: list[str],
        workdir: Path,
        pi_home: Path,
        example_id: str,
        lines: list[str],
        prompt: str,
    ) -> tuple[int, str]:
        """Run the CLI with no time limit of its own; the runner's per-file deadline cancels it."""
        env = {**os.environ, "PI_CODING_AGENT_DIR": str(pi_home)}

        # stderr -> file, never an undrained PIPE (see claude_code_extract).
        stderr_path = pi_home.parent / "pi_stderr.log"
        try:
            errf: Any = open(stderr_path, "w", encoding="utf-8")
        except OSError:
            errf = None

        try:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(workdir),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=errf if errf is not None else subprocess.DEVNULL,
                    text=True,
                    env=env,
                )
            except FileNotFoundError as e:
                raise ProviderConfigError(
                    f"pi CLI not found (tried '{self._pi_bin}'). Install "
                    f"@earendil-works/pi-coding-agent and ensure it is on PATH: {e}"
                ) from e

            with self._procs_lock:
                self._procs[example_id] = proc

            # Hand pi the prompt and close stdin so it stops waiting for more.
            if proc.stdin is not None:
                try:
                    proc.stdin.write(prompt)
                except BrokenPipeError:
                    pass  # pi died early; the returncode check below reports it
                finally:
                    proc.stdin.close()

            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\n")
                    if line:
                        lines.append(line)
                proc.wait()
            finally:
                self._terminate(proc)
                with self._procs_lock:
                    self._procs.pop(example_id, None)

            return int(proc.returncode if proc.returncode is not None else 0), self._read_tail(stderr_path)
        finally:
            if errf is not None:
                errf.close()

    @staticmethod
    def _read_tail(path: Path, max_chars: int = 4000) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-max_chars:]

    @staticmethod
    def _terminate(proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _raise_for_status(self, returncode: int, summary: PiEventSummary, stderr_tail: str = "") -> None:
        error_messages = summary.error_messages
        if returncode == 0 and not error_messages:
            return
        message = error_messages[-1] if error_messages else (stderr_tail or f"pi exited {returncode}")
        lowered = message.lower()
        if any(keyword in lowered for keyword in _RATE_LIMIT_KEYWORDS):
            raise ProviderRateLimitError(f"pi_extract rate-limited: {message}")
        if any(keyword in lowered for keyword in _TRANSIENT_KEYWORDS):
            raise ProviderTransientError(f"pi_extract transient CLI error: {message}")
        raise ProviderPermanentError(f"pi_extract CLI exited {returncode}: {message}")

    @staticmethod
    def _read_output_json(output_path: Path) -> dict[str, Any]:
        if not output_path.exists():
            raise ProviderPermanentError("pi_extract: CLI completed but final output JSON was not written.")
        try:
            data = json.loads(output_path.read_text())
        except json.JSONDecodeError as e:
            raise ProviderPermanentError(f"pi_extract: final output is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ProviderPermanentError(f"pi_extract: final output must be a JSON object, got {type(data).__name__}.")
        return data

    def _pricing_snapshot(self) -> dict[str, Any]:
        return {
            "pricing_basis": "pi_reported_cost_from_models_json_rates",
            "input_price_per_1m": self._input_price_per_1m,
            "output_price_per_1m": self._output_price_per_1m,
            "cache_read_price_per_1m": self._cache_read_price_per_1m,
        }

    def _config_snapshot(self) -> dict[str, Any]:
        return {
            "provider": "pi_extract",
            HARNESS_VERSION: self._harness.version,
            "model": self._model,
            "base_url": self._base_url,
            "wire_api": self._wire_api,
            "provider_label": self._provider_label,
            "max_cost_usd": self._max_cost_usd,
            "max_tokens": self._max_tokens,
            "context_window": self._context_window,
            "promote_repeated_structure": self._promote_repeated,
            "additional_properties_false": self._additional_properties_false,
            "all_properties_required": self._all_properties_required,
            "evidence_mode": self._evidence_mode,
            "sampling_params": self._sampling_params,
            "extra_instructions": self._extra_instructions,
            "enable_vision": self._enable_vision,
            **self._pricing_snapshot(),
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"{type(self).__name__} only supports EXTRACT product type, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override."
            )

        src = Path(request.source_file_path)
        if not src.exists():
            raise ProviderPermanentError(f"File not found: {src}")
        ext = src.suffix.lower()
        if ext != ".pdf":
            raise ProviderPermanentError(f"{type(self).__name__} currently supports PDFs only, got {src.suffix}")

        started_at = datetime.now()
        harness = self._harness.prepare(pipeline, request)
        schema = self._prepare_schema(request.schema_override)
        num_pages = page_count(src)

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            staged_name = f"input{ext}"
            output_path = workdir / "output.json"
            shutil.copy2(src, workdir / staged_name)

            # Per-request pi home beside the workdir: concurrent requests never
            # share provider config and the agent never sees the key file.
            pi_home = Path(tmp) / "pi-home" / "agent"
            pi_home.mkdir(parents=True)
            (pi_home / "models.json").write_text(json.dumps(self._build_models_json(), indent=2), encoding="utf-8")

            prompt = self._build_prompt(schema, staged_name)
            cmd = self._build_cmd()
            cmd[0] = str(harness["executable"])
            execution_budget().remaining()

            lines: list[str] = []
            returncode: int | None = None
            stderr_tail = ""
            agent_citations: Any = None
            data: dict[str, Any] = {}
            summary: PiEventSummary | None = None
            accounting: dict[str, Any] | None = None
            try:
                returncode, stderr_tail = self._run_cli(cmd, workdir, pi_home, request.example_id, lines, prompt)
                summary = summarize_pi_events(lines)
                accounting = record_response_cost(
                    summary.cost["total"] if summary.cost_observed else None,
                    summary.usage,
                    complete=summary.cost_complete and returncode == 0 and len(summary.error_messages) == 0,
                )
                self._raise_for_status(returncode, summary, stderr_tail)
                data = self._read_output_json(output_path)
                if self._evidence_mode:
                    agent_citations = read_agent_citations_file(workdir / "citations.json")
            except ProviderError as exc:
                summary = summary or summarize_pi_events(lines)
                payload: dict[str, Any] = {
                    "returncode": returncode,
                    "stderr_tail": stderr_tail,
                    "trace_jsonl_lines": lines,
                    "trace_tail": summary.trace_tail,
                    "tool_execution": summary.tool_diagnostics,
                    "config": self._config_snapshot(),
                    "cmd": cmd,
                }
                if output_path.exists():
                    payload["output_text_tail"] = self._read_tail(output_path, max_chars=100_000)
                    try:
                        payload["output_json"] = json.loads(output_path.read_text())
                    except json.JSONDecodeError:
                        pass
                if isinstance(exc.debug_payload, dict):
                    payload.update(exc.debug_payload)
                exc.debug_payload = payload
                if accounting is None:
                    accounting = record_response_cost(
                        summary.cost["total"] if summary.cost_observed else None,
                        summary.usage,
                        complete=False,
                    )
                with preserve_cost_on_error({**accounting, USAGE: summary.usage}):
                    raise

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        assert summary is not None and accounting is not None
        cost_usd = accounting[COST_USD]
        raw_output: dict[str, Any] = {
            "data": data,
            "harness": harness,
            "model": self._model,
            USAGE: summary.usage,
            "num_pages": num_pages,
            **accounting,
            COST_PER_PAGE_USD: (cost_usd / num_pages) if cost_usd is not None and num_pages else None,
            "pricing": self._pricing_snapshot(),
            "cost_exceeded_budget": (
                self._max_cost_usd is not None and cost_usd is not None and cost_usd > self._max_cost_usd
            ),
            "_config": self._config_snapshot(),
            "_tool_execution": summary.tool_diagnostics,
            "_trace_tail": summary.trace_tail,
            "_pi_cost_breakdown": summary.cost,
            "_pi_artifacts": {
                "output_json": data,
                "trace_jsonl_lines": lines,
                "stderr_tail": stderr_tail,
            },
        }

        if self._evidence_mode:
            citations, evidence_stats = citations_from_agent_file(agent_citations, source="pi_extract_evidence")
            raw_output["field_citations"] = [citation.model_dump() for citation in citations]
            raw_output["evidence_stats"] = evidence_stats.as_dict()

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

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        return normalize_extract_result(raw_result)

    def cancel(self, example_id: str) -> bool:
        probe_cancelled = self._harness.cancel(example_id)
        with self._procs_lock:
            proc = self._procs.get(example_id)
        if proc is None:
            return probe_cancelled
        self._terminate(proc)
        return True
