"""Agentic extraction provider that shells out to Tencent's CodeBuddy Code CLI (`codebuddy -p`).

Same harness shape as ``claude_code_extract`` / ``codex_code_extract``: each
document is copied into an isolated temporary workdir with ``./schema.json``,
the agent is asked to inspect that local file with shell tools and write
``./output.json`` (and ``./citations.json`` in evidence mode), and the result is
normalized through the standard Extract output path.

Each request gets a private ``HOME`` beside the workdir with
``.codebuddy/models.json`` pointing at the configured OpenAI-compatible
chat-completions endpoint. The key is written as an environment-variable
reference (``${TENCENT_TOKENHUB_API_KEY}``) that CodeBuddy expands at startup,
so the file never holds the key and the user's real ``~/.codebuddy`` is never
touched.

CodeBuddy reports token usage but no dollar cost, so cost is computed from
the usage counters at the per-million rates in the pipeline config. Its
``input_tokens`` already include cached reads, which are billed at the cache
rate; ``output_tokens`` include reasoning tokens.

Each document gets one CLI run with no time limit of its own; the runner's
per-file deadline cancels it through ``cancel``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from extract_bench.inference.cost_accounting import preserve_cost_on_error, record_response_cost
from extract_bench.inference.cost_contract import COST_PER_PAGE_USD, COST_USD, USAGE
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
from extract_bench.inference.providers.extract.cli_harness import CLAUDE_UPDATE_ENV, HARNESS_VERSION, CliHarness
from extract_bench.inference.providers.extract.direct_model_utils import (
    IMAGE_EXTENSIONS,
    add_additional_properties_false,
    force_all_properties_required,
    normalize_extract_result,
    page_count,
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
_MAX_SHELL_CMD_CHARS = 2000
# Above this size the prompt points at ./schema.json instead of inlining the schema.
_MAX_INLINE_PROMPT_BYTES = 120_000
_SCHEMA_FILE_NAME = "schema.json"
_TURN_USAGE_KEY = "codebuddy.ai/turnUsage"

_BILLING_KEYWORDS = ("402", "postpaid billing", "free trial quota", "insufficient_quota")
_RATE_LIMIT_KEYWORDS = ("rate limit", "rate_limit", "429", "quota")
_TRANSIENT_KEYWORDS = (
    "overloaded",
    "high demand",
    "529",
    "503",
    "502",
    "504",
    "500",
    "timeout",
    "timed out",
    "econnreset",
    "econnrefused",
    "socket hang up",
)
# Host settings that would point CodeBuddy at another model or account.
_EXCLUDED_ENV_VARS = frozenset(
    {"PWD", "CODEBUDDY_API_KEY", "CODEBUDDY_AUTH_TOKEN", "CODEBUDDY_BASE_URL", "CODEBUDDY_MODEL"}
)
_PRICE_KEYS = ("input_price_per_1m", "output_price_per_1m", "cache_read_price_per_1m")
_EMPTY_USAGE = {
    "input_tokens": 0,
    "uncached_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}


def _normalize_usage(raw_usage: Mapping[str, Any]) -> dict[str, int]:
    input_tokens = int(raw_usage.get("input_tokens", 0) or 0)
    cache_read = int(raw_usage.get("cache_read_input_tokens", 0) or 0)
    output_tokens = int(raw_usage.get("output_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "uncached_input_tokens": max(0, input_tokens - cache_read),
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": int(raw_usage.get("cache_creation_input_tokens", 0) or 0),
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _sum_stage_usage(stages: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    totals = {
        key: sum(int(stage.get(key, 0) or 0) for stage in stages.values())
        for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")
    }
    return _normalize_usage(totals)


def _stage_update(event: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    """Return ``(stage id, usage)`` for a per-stage ``usage_update`` event."""
    if event.get("type") != "system" or event.get("subtype") != "usage_update":
        return None
    turn_usage = (event.get("_meta") or {}).get(_TURN_USAGE_KEY)
    if not isinstance(turn_usage, dict) or turn_usage.get("kind") != "stage":
        return None
    stage = turn_usage.get("stage")
    if isinstance(stage, dict) and isinstance(stage.get("id"), str) and isinstance(stage.get("usage"), dict):
        return stage["id"], stage["usage"]
    return None


def usage_from_events(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], bool, bool]:
    """Return ``(usage, observed, complete)`` from a CodeBuddy stream-json trace.

    The terminal ``result`` event carries the run total. Without it (a killed or
    failed run), per-stage ``usage_update`` events are deduplicated by stage id
    and summed, falling back to the last running ``snapshot`` total; that usage
    is observed but not complete.
    """
    result_event = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if result_event is not None and isinstance(result_event.get(USAGE), dict):
        return _normalize_usage(result_event[USAGE]), True, not bool(result_event.get("is_error"))

    stages: dict[str, Mapping[str, Any]] = {}
    snapshot: Mapping[str, Any] | None = None
    for event in events:
        update = _stage_update(event)
        if update is not None:
            stages[update[0]] = update[1]
            continue
        if event.get("type") == "system" and event.get("subtype") == "usage_update":
            turn_usage = (event.get("_meta") or {}).get(_TURN_USAGE_KEY)
            if isinstance(turn_usage, dict) and turn_usage.get("kind") == "snapshot":
                total = (turn_usage.get("total") or {}).get(USAGE)
                if isinstance(total, dict):
                    snapshot = total
    if stages:
        return _sum_stage_usage(stages), True, False
    if snapshot is not None:
        return _normalize_usage(snapshot), True, False
    return dict(_EMPTY_USAGE), False, False


def tool_usage_from_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tool_counts: dict[str, int] = {}
    bash_commands: list[str] = []
    failed = 0
    for event in events:
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if event.get("type") == "assistant" and block.get("type") == "tool_use":
                name = str(block.get("name") or "")
                tool_counts[name] = tool_counts.get(name, 0) + 1
                if name == "Bash":
                    command = str((block.get("input") or {}).get("command") or "")
                    if len(command) > _MAX_SHELL_CMD_CHARS:
                        command = command[:_MAX_SHELL_CMD_CHARS] + f"...[+{len(command) - _MAX_SHELL_CMD_CHARS} chars]"
                    bash_commands.append(command)
            elif event.get("type") == "user" and block.get("type") == "tool_result" and block.get("is_error"):
                failed += 1
    return {"tool_counts": tool_counts, "bash_commands": bash_commands, "failed_tool_calls": failed}


def _parse_json_lines(lines: Sequence[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _trace_tail(lines: Sequence[str]) -> list[Any]:
    tail: list[Any] = []
    for line in lines[-_TRACE_TAIL_EVENTS:]:
        try:
            tail.append(json.loads(line))
        except json.JSONDecodeError:
            tail.append(line)
    return tail


def _read_tail(path: Path, max_chars: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""


def _terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _classify_error(returncode: int, message: str) -> ProviderError:
    lowered = message.lower()
    if any(keyword in lowered for keyword in _BILLING_KEYWORDS):
        return ProviderPermanentError(f"codebuddy_extract quota/billing error: {message}")
    if any(keyword in lowered for keyword in _RATE_LIMIT_KEYWORDS):
        return ProviderRateLimitError(f"codebuddy_extract rate-limited: {message}")
    if any(keyword in lowered for keyword in _TRANSIENT_KEYWORDS):
        return ProviderTransientError(f"codebuddy_extract transient CLI error: {message}")
    return ProviderPermanentError(f"codebuddy_extract CLI error (exit={returncode}): {message}")


@register_provider("codebuddy_extract")
class CodeBuddyExtractProvider(Provider):
    """Agentic extraction by shelling out to the CodeBuddy Code CLI (`codebuddy -p`)."""

    DEFAULT_VENDOR = "Tencent TokenHub"
    DEFAULT_API_KEY_ENV = "TENCENT_TOKENHUB_API_KEY"
    DEFAULT_MAX_INPUT_TOKENS = 960_000
    DEFAULT_MAX_OUTPUT_TOKENS = 64_000
    PRICING_BASIS = "token_usage_at_configured_rates"
    # One CLI run per document: a retry would start a new paid agent session.
    retries_transient_errors = False

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None) -> None:
        super().__init__(provider_name, base_config)
        missing = [key for key in ("model", "url", *_PRICE_KEYS) if key not in self.base_config]
        if missing:
            raise ProviderConfigError(f"codebuddy_extract requires pipeline config keys: {', '.join(missing)}")
        self._model: str = str(self.base_config["model"])
        # Full chat-completions URL; CodeBuddy posts to it as given.
        self._url: str = str(self.base_config["url"])
        self._vendor: str = str(self.base_config.get("vendor", self.DEFAULT_VENDOR))
        self._api_key_env: str = str(self.base_config.get("api_key_env") or self.DEFAULT_API_KEY_ENV)
        self._api_key: str | None = self.base_config.get("api_key") or os.getenv(self._api_key_env)
        self._codebuddy_bin: str = str(self.base_config.get("codebuddy_bin", "codebuddy"))
        self._harness = CliHarness("codebuddy_extract", self._codebuddy_bin, self.base_config.get(HARNESS_VERSION))
        # Enforced while the CLI runs: the process is stopped once the running
        # cost passes this amount. None disables the limit.
        self._max_cost_usd: float | None = self.base_config.get("max_cost_usd")
        self._max_input_tokens: int = int(self.base_config.get("max_input_tokens", self.DEFAULT_MAX_INPUT_TOKENS))
        self._max_output_tokens: int = int(self.base_config.get("max_output_tokens", self.DEFAULT_MAX_OUTPUT_TOKENS))
        # Declares image input in models.json so the agent can read rendered pages.
        self._enable_vision: bool = bool(self.base_config.get("enable_vision", True))
        self._extra_flags: list[str] = list(self.base_config.get("extra_flags", []))
        self._extra_instructions: str = str(self.base_config.get("extra_instructions") or "").strip()
        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))
        self._all_properties_required: bool = bool(self.base_config.get("all_properties_required", False))
        # Evidence mode: the agent also writes ./citations.json with a page and
        # box per extracted value (same side file as claude_code_extract).
        self._evidence_mode: bool = bool(self.base_config.get("evidence_mode", False))

        self._input_price_per_1m: float = float(self.base_config["input_price_per_1m"])
        self._cache_read_price_per_1m: float = float(self.base_config["cache_read_price_per_1m"])
        self._output_price_per_1m: float = float(self.base_config["output_price_per_1m"])

        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._procs_lock = threading.Lock()

    def _prepare_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if self._additional_properties_false:
            schema = add_additional_properties_false(schema)
        if self._all_properties_required:
            schema = force_all_properties_required(schema)
        return schema

    def _build_prompt(self, schema: dict[str, Any], staged_name: str, *, schema_file: str | None = None) -> str:
        if schema_file is None:
            schema_section = (
                "Return a single JSON object conforming to this schema as your final answer:\n\n"
                f"```json\n{json.dumps(schema, indent=2)}\n```\n\n"
            )
        else:
            schema_section = (
                f"Return a single JSON object conforming to the JSON Schema in ./{schema_file} as your final answer. "
                "The schema is too large to include here; read it from that file.\n\n"
            )
        evidence_rule = f"\n{agent_evidence_instruction()}" if self._evidence_mode else ""
        extra = f"\n\n{self._extra_instructions}" if self._extra_instructions else ""
        return (
            f"Extract structured data from the document file `./{staged_name}` in the current directory.\n\n"
            "Use only local file inspection and shell commands. Do not use web search, network calls, "
            "browser tools, or external services. Temporary scratch files inside the current directory are OK.\n\n"
            f"{schema_section}"
            "Rules:\n"
            "- Use null for fields not present in the document.\n"
            "- For list/array fields, enumerate every relevant row visible in the document; never collapse rows.\n"
            "- For large regular tables, prefer writing and running a local script to parse/enumerate rows.\n"
            "- For forms, prefer direct field extraction from the document content.\n"
            "- Write the resulting JSON object to ./output.json and validate that it is valid JSON before stopping.\n"
            f"- Do not print the JSON to your assistant output.{evidence_rule}{extra}"
        )

    def _stage_prompt(self, schema: dict[str, Any], staged_name: str, workdir: Path) -> tuple[str, str]:
        """Write ``./schema.json`` and return ``(prompt, schema_delivery)``."""
        (workdir / _SCHEMA_FILE_NAME).write_text(json.dumps(schema, indent=2), encoding="utf-8")
        prompt = self._build_prompt(schema, staged_name)
        if len(prompt.encode("utf-8")) <= _MAX_INLINE_PROMPT_BYTES:
            return prompt, "inline"
        return self._build_prompt(schema, staged_name, schema_file=_SCHEMA_FILE_NAME), "file"

    def _build_models_json(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "id": self._model,
                    "name": self._model,
                    "vendor": self._vendor,
                    "url": self._url,
                    "apiKey": f"${{{self._api_key_env}}}",
                    "maxInputTokens": self._max_input_tokens,
                    "maxOutputTokens": self._max_output_tokens,
                    "supportsToolCall": True,
                    "supportsImages": self._enable_vision,
                }
            ],
            "availableModels": [self._model],
        }

    def _build_cmd(self, mcp_config: Path) -> list[str]:
        # The prompt goes over stdin. An empty MCP config with --strict-mcp-config
        # and user-only settings keep host tools and project settings out of the run.
        return [
            self._codebuddy_bin,
            "-p",
            "-y",
            "--model",
            self._model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--strict-mcp-config",
            "--mcp-config",
            str(mcp_config),
            "--setting-sources",
            "user",
            "--disallowedTools",
            "WebSearch,WebFetch",
            "--no-session-persistence",
            *self._extra_flags,
        ]

    def _subprocess_env(self, home: Path, workdir: Path) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if key not in _EXCLUDED_ENV_VARS}
        env.update(
            {
                **CLAUDE_UPDATE_ENV,
                "HOME": str(home),
                "PWD": str(workdir),
                "CODEBUDDY_IS_SANDBOX": "1",
                # Helper and subagent calls use the same model as the main agent.
                "CODEBUDDY_SMALL_FAST_MODEL": self._model,
                "CODEBUDDY_BIG_SLOW_MODEL": self._model,
                "CODEBUDDY_CODE_SUBAGENT_MODEL": self._model,
                self._api_key_env: self._api_key or "",
            }
        )
        return env

    def _estimate_cost(self, usage: Mapping[str, Any]) -> float:
        return (
            int(usage.get("uncached_input_tokens", 0) or 0) * self._input_price_per_1m
            + int(usage.get("cache_read_input_tokens", 0) or 0) * self._cache_read_price_per_1m
            + int(usage.get("output_tokens", 0) or 0) * self._output_price_per_1m
        ) / 1_000_000

    def _run_cli(
        self,
        cmd: list[str],
        workdir: Path,
        home: Path,
        example_id: str,
        lines: list[str],
        prompt: str,
    ) -> tuple[int, str]:
        """Run the CLI with no time limit of its own; the runner's per-file deadline cancels it."""
        stderr_path = home.parent / "codebuddy_stderr.log"
        with open(stderr_path, "w", encoding="utf-8") as errf:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(workdir),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=errf,
                    text=True,
                    env=self._subprocess_env(home, workdir),
                )
            except FileNotFoundError as e:
                raise ProviderConfigError(
                    f"codebuddy CLI not found (tried '{self._codebuddy_bin}'). "
                    f"Install @tencent-ai/codebuddy-code and ensure it is on PATH: {e}"
                ) from e

            with self._procs_lock:
                self._procs[example_id] = proc

            if proc.stdin is not None:
                try:
                    proc.stdin.write(prompt)
                except BrokenPipeError:
                    pass  # the CLI died early; the status check below reports it
                finally:
                    proc.stdin.close()

            stages: dict[str, Mapping[str, Any]] = {}
            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\n")
                    if not line:
                        continue
                    lines.append(line)
                    if self._max_cost_usd is None:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    update = _stage_update(event) if isinstance(event, dict) else None
                    if update is None:
                        continue
                    stages[update[0]] = update[1]
                    running_cost = self._estimate_cost(_sum_stage_usage(stages))
                    if running_cost > self._max_cost_usd:
                        _terminate(proc)
                        raise ProviderPermanentError(
                            f"codebuddy_extract exceeded max_cost_usd ${self._max_cost_usd:.4f} "
                            f"(spent ${running_cost:.4f}); stopped mid-run."
                        )
                proc.wait()
            finally:
                _terminate(proc)
                with self._procs_lock:
                    self._procs.pop(example_id, None)

        return int(proc.returncode if proc.returncode is not None else 0), _read_tail(stderr_path)

    @staticmethod
    def _raise_for_status(returncode: int, events: Sequence[Mapping[str, Any]], stderr_tail: str) -> None:
        result_event = next((e for e in reversed(events) if e.get("type") == "result"), None)
        if result_event is not None and not result_event.get("is_error") and returncode == 0:
            return
        if result_event is None:
            message = stderr_tail or f"no terminal 'result' event (exit={returncode})"
        else:
            errors = result_event.get("errors")
            detail = errors[-1] if isinstance(errors, list) and errors else None
            message = str(detail or result_event.get("result") or stderr_tail or f"exit={returncode}")
        raise _classify_error(returncode, message)

    @staticmethod
    def _read_output_json(output_path: Path) -> dict[str, Any]:
        if not output_path.exists():
            raise ProviderPermanentError("codebuddy_extract: CLI completed but ./output.json was not written.")
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ProviderPermanentError(f"codebuddy_extract: ./output.json is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ProviderPermanentError(
                f"codebuddy_extract: ./output.json must be a JSON object, got {type(data).__name__}."
            )
        return data

    def _pricing_snapshot(self) -> dict[str, Any]:
        return {
            "pricing_basis": self.PRICING_BASIS,
            "input_price_per_1m": self._input_price_per_1m,
            "cache_read_price_per_1m": self._cache_read_price_per_1m,
            "output_price_per_1m": self._output_price_per_1m,
        }

    def _config_snapshot(self) -> dict[str, Any]:
        return {
            "provider": "codebuddy_extract",
            HARNESS_VERSION: self._harness.version,
            "model": self._model,
            "url": self._url,
            "vendor": self._vendor,
            "max_cost_usd": self._max_cost_usd,
            "max_input_tokens": self._max_input_tokens,
            "max_output_tokens": self._max_output_tokens,
            "enable_vision": self._enable_vision,
            "additional_properties_false": self._additional_properties_false,
            "all_properties_required": self._all_properties_required,
            "evidence_mode": self._evidence_mode,
            "extra_instructions": self._extra_instructions,
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
        if ext != ".pdf" and ext not in IMAGE_EXTENSIONS:
            raise ProviderPermanentError(f"{type(self).__name__} supports PDFs and images, got {src.suffix}")
        if not self._api_key:
            raise ProviderConfigError(f"{self._api_key_env} is required for codebuddy_extract.")

        started_at = datetime.now()
        harness = self._harness.prepare(pipeline, request)
        schema = self._prepare_schema(request.schema_override)
        num_pages = page_count(src)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            workdir = root / "work"
            home = root / "home"
            workdir.mkdir()
            (home / ".codebuddy").mkdir(parents=True)
            models_path = home / ".codebuddy" / "models.json"
            models_path.write_text(json.dumps(self._build_models_json(), indent=2), encoding="utf-8")
            models_path.chmod(0o600)
            mcp_config = home / "mcp_config.json"
            mcp_config.write_text('{"mcpServers": {}}', encoding="utf-8")

            staged_name = f"input{ext}"
            output_path = workdir / "output.json"
            shutil.copy2(src, workdir / staged_name)
            prompt, schema_delivery = self._stage_prompt(schema, staged_name, workdir)
            cmd = self._build_cmd(mcp_config)
            cmd[0] = str(harness["executable"])
            execution_budget().remaining()

            lines: list[str] = []
            returncode: int | None = None
            stderr_tail = ""
            agent_citations: Any = None
            data: dict[str, Any] = {}
            events: list[dict[str, Any]] = []
            accounting: dict[str, Any] | None = None
            usage: dict[str, int] = dict(_EMPTY_USAGE)
            try:
                returncode, stderr_tail = self._run_cli(cmd, workdir, home, request.example_id, lines, prompt)
                events = _parse_json_lines(lines)
                usage, observed, complete = usage_from_events(events)
                accounting = record_response_cost(
                    self._estimate_cost(usage) if observed else None,
                    usage,
                    complete=complete and returncode == 0,
                )
                self._raise_for_status(returncode, events, stderr_tail)
                data = self._read_output_json(output_path)
                if self._evidence_mode:
                    agent_citations = read_agent_citations_file(workdir / "citations.json")
            except ProviderError as exc:
                events = events or _parse_json_lines(lines)
                payload: dict[str, Any] = {
                    "returncode": returncode,
                    "stderr_tail": stderr_tail,
                    "trace_tail": _trace_tail(lines),
                    "config": self._config_snapshot(),
                    "cmd": cmd,
                }
                if isinstance(exc.debug_payload, dict):
                    payload.update(exc.debug_payload)
                exc.debug_payload = payload
                if accounting is None:
                    usage, observed, _ = usage_from_events(events)
                    accounting = record_response_cost(
                        self._estimate_cost(usage) if observed else None, usage, complete=False
                    )
                with preserve_cost_on_error({**accounting, USAGE: usage}):
                    raise

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        assert accounting is not None
        cost_usd = accounting[COST_USD]
        raw_output: dict[str, Any] = {
            "data": data,
            "harness": harness,
            "model": self._model,
            USAGE: usage,
            "num_pages": num_pages,
            **accounting,
            COST_PER_PAGE_USD: (cost_usd / num_pages) if cost_usd is not None and num_pages else None,
            "pricing": self._pricing_snapshot(),
            "cost_exceeded_budget": (
                self._max_cost_usd is not None and cost_usd is not None and cost_usd > self._max_cost_usd
            ),
            "schema_delivery": schema_delivery,
            "_config": self._config_snapshot(),
            **tool_usage_from_events(events),
            "_trace_tail": _trace_tail(lines),
        }

        if self._evidence_mode:
            citations, evidence_stats = citations_from_agent_file(agent_citations, source="codebuddy_extract_evidence")
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
        _terminate(proc)
        return True
