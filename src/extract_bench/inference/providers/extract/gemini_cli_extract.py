"""Agentic extraction provider that shells out to the Gemini CLI (`gemini -p`).

Same harness shape as ``claude_code_extract`` / ``pi_extract``: the document is
staged in a temp workdir, the agent reads it with its own tools (``read_file``
sends a PDF to the model natively) and writes ``./output.json`` against the
benchmark schema.

Each request gets its own ``GEMINI_CLI_HOME`` with a generated ``settings.json``
(gateway API-key auth, web tools excluded, telemetry off), so the user's ``~/.gemini``
hooks, extensions and OAuth session never leak into a run and concurrent
requests never share state. ``thinking_level`` is pinned through a
``modelConfigs.customOverrides`` entry: left alone, the CLI picks its own
thinking config per model alias.

Native HTTP requests and streamed responses are journaled through a loopback
recorder forwarding to the configured native Gemini endpoint. Costs use per-request usage as it arrives,
including hidden thinking tokens; missing usage remains explicitly unknown.
The CLI terminal stats are retained for comparison, never added a second time.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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
from extract_bench.inference.providers.extract.direct_model_utils import (
    IMAGE_EXTENSIONS,
    add_additional_properties_false,
    normalize_extract_result,
    page_count,
    promote_repeated_structure,
)
from extract_bench.inference.providers.extract.gemini_cli_trace import GeminiCliTrace
from extract_bench.inference.providers.extract.gemini_cli_transport import GeminiCliTransport
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType
from extract_bench.utils.gemini_pricing import cost_from_request_usage

_REQUEST_RETRY_PATTERN = re.compile(r"Attempt \d+ failed[^\n]*Retrying with backoff")
_TRACE_TAIL_EVENTS = 20
_MAX_SHELL_CMD_CHARS = 2000

_RATE_LIMIT_KEYWORDS = ("rate limit", "rate_limit", "429", "quota", "resource_exhausted")
_TRANSIENT_KEYWORDS = ("overloaded", "high demand", "unavailable", "500", "502", "503", "504", "timeout", "timed out")


@register_provider("gemini_cli_extract")
class GeminiCliExtractProvider(Provider):
    """Agentic extraction by shelling out to the Gemini CLI (`gemini -p`)."""

    DEFAULT_MODEL = "gemini-4-argon"
    retries_transient_errors = False  # one run per document, like claude_code_extract

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._model: str = self.base_config.get("model", self.DEFAULT_MODEL)
        self._thinking_level: str | None = self.base_config.get("thinking_level")
        self._max_cost_usd: float | None = self.base_config.get("max_cost_usd")
        self._gemini_bin: str = self.base_config.get("gemini_bin", "gemini")
        self._promote_repeated: bool = bool(self.base_config.get("promote_repeated_structure", True))
        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))
        self._evidence_mode: bool = bool(self.base_config.get("evidence_mode", False))
        self._base_url = os.environ.get("GOOGLE_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip(
            "/"
        )
        endpoint = urlsplit(self._base_url)
        if (
            endpoint.scheme not in {"http", "https"}
            or endpoint.hostname is None
            or any(value is not None for value in (endpoint.username, endpoint.password))
            or endpoint.query != ""
            or endpoint.fragment != ""
        ):
            raise ProviderConfigError(
                "GOOGLE_GEMINI_BASE_URL must be an HTTP(S) URL without credentials, query or fragment"
            )
        self._api_key = next(
            (
                os.environ[name]
                for name in ("GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
                if os.environ.get(name, "") != ""
            ),
            "",
        )
        if self._api_key == "":
            raise ProviderConfigError("Set GOOGLE_GEMINI_API_KEY, GEMINI_API_KEY or GOOGLE_API_KEY")
        self._headers: dict[str, str] = {}
        for entry in re.split(r",(?=\s*[^,:]+:)", os.environ.get("GEMINI_CLI_CUSTOM_HEADERS", "")):
            if entry.strip() == "":
                continue
            name, separator, value = entry.partition(":")
            name, value = name.strip().lower(), value.strip()
            if separator == "" or name in {"host", "content-length", "transfer-encoding", "connection"}:
                raise ProviderConfigError("Invalid GEMINI_CLI_CUSTOM_HEADERS forwarding header")
            self._headers[name] = value
        self._traces: dict[str, GeminiCliTrace] = {}
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._procs_lock = threading.Lock()

    def _prepare_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if self._promote_repeated:
            schema = promote_repeated_structure(schema)
        if self._additional_properties_false:
            schema = add_additional_properties_false(schema)
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
        )

    def _build_settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = {
            "security": {"auth": {"selectedType": "gemini-api-key"}},
            "general": {"disableAutoUpdate": True, "disableUpdateNag": True},
            "privacy": {"usageStatisticsEnabled": False},
            "telemetry": {"enabled": False},
            "tools": {"exclude": ["google_web_search", "web_fetch"]},
        }
        if self._thinking_level is not None:
            settings["modelConfigs"] = {
                "customOverrides": [
                    {
                        "match": {"model": self._model},
                        "modelConfig": {
                            "generateContentConfig": {"thinkingConfig": {"thinkingLevel": self._thinking_level.upper()}}
                        },
                    }
                ]
            }
        return settings

    def _build_cmd(self) -> list[str]:
        # The prompt goes over stdin (`-p` appends to it): it embeds the whole
        # schema and a large one would push argv past ARG_MAX.
        return [
            self._gemini_bin,
            "--model",
            self._model,
            "--output-format",
            "stream-json",
            "--approval-mode",
            "yolo",
            "--skip-trust",
            "-p",
            "",
        ]

    def _run_cli(
        self,
        cmd: list[str],
        workdir: Path,
        home: Path,
        example_id: str,
        lines: deque[str],
        prompt: str,
        *,
        base_url: str,
        trace: GeminiCliTrace,
        tool_usage: dict[str, Any],
    ) -> tuple[int, str]:
        """Run the CLI with no time limit of its own; the runner's per-file timeout cancels it."""
        # Personal Google credentials and inherited endpoint/header overrides must
        # never bypass the recorder or change the configured upstream route.
        excluded = {
            "GOOGLE_API_KEY",
            "GOOGLE_GEMINI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_GEMINI_BASE_URL",
            "GOOGLE_VERTEX_BASE_URL",
            "GEMINI_CLI_CUSTOM_HEADERS",
            "GEMINI_API_KEY_AUTH_MECHANISM",
        }
        env = {k: v for k, v in os.environ.items() if k not in excluded}
        env.update(
            {"GEMINI_CLI_HOME": str(home), "GEMINI_API_KEY": "local-recorder", "GOOGLE_GEMINI_BASE_URL": base_url}
        )

        for name in ("NO_PROXY", "no_proxy"):
            env[name] = ",".join(filter(None, [env.get(name), "127.0.0.1", "localhost"]))

        # stderr -> file, never an undrained PIPE (see claude_code_extract).
        stderr_path = home / "gemini_stderr.log"
        with open(stderr_path, "w", encoding="utf-8") as errf:
            if trace.cancelled.is_set():
                raise ProviderPermanentError("Gemini CLI extraction cancelled before process start")
            trace.emit("cli.started", command=cmd, prompt=prompt, settings=self._build_settings())
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(workdir),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=errf,
                    text=True,
                    env=env,
                )
            except FileNotFoundError as e:
                raise ProviderConfigError(
                    f"gemini CLI not found (tried '{self._gemini_bin}'). Install @google/gemini-cli: {e}"
                ) from e

            with self._procs_lock:
                self._procs[example_id] = proc
                if trace.cancelled.is_set():
                    self._terminate(proc)
            try:
                assert proc.stdin is not None and proc.stdout is not None
                try:
                    proc.stdin.write(prompt)
                except BrokenPipeError:
                    pass  # the CLI died early; the exit code reports it
                finally:
                    proc.stdin.close()
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\n")
                    if line != "":
                        lines.append(line)
                        trace.emit("cli.stdout", line=line)
                        for event in self._events([line]):
                            usage = self._tool_usage([event])
                            for name, count in usage["tool_counts"].items():
                                tool_usage["tool_counts"][name] = tool_usage["tool_counts"].get(name, 0) + count
                            tool_usage["shell_commands"].extend(usage["shell_commands"])
                            tool_usage["failed_tool_calls"] += usage["failed_tool_calls"]
                proc.wait()
            finally:
                self._terminate(proc)
                with self._procs_lock:
                    self._procs.pop(example_id, None)

        stderr_tail = self._read_tail(stderr_path)
        trace.emit("cli.finished", returncode=proc.returncode, stderr_tail=stderr_tail)
        return int(proc.returncode or 0), stderr_tail

    @staticmethod
    def _read_tail(path: Path, max_chars: int = 4000) -> str:
        try:
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - max_chars * 4))
                return stream.read().decode("utf-8", errors="replace")[-max_chars:]
        except OSError:
            return ""

    @staticmethod
    def _terminate(proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    @staticmethod
    def _events(lines: Sequence[str]) -> list[dict[str, Any]]:
        events = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    @staticmethod
    def _tool_usage(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Tool calls (never their outputs) and every shell command, capped in length."""
        tool_counts: dict[str, int] = {}
        shell_commands: list[str] = []
        failed = 0
        for event in events:
            if event.get("type") == "tool_result" and event.get("status") != "success":
                failed += 1
            if event.get("type") != "tool_use":
                continue
            name = str(event.get("tool_name") or "")
            tool_counts[name] = tool_counts.get(name, 0) + 1
            if name == "run_shell_command":
                command = str((event.get("parameters") or {}).get("command") or "")
                if len(command) > _MAX_SHELL_CMD_CHARS:
                    command = command[:_MAX_SHELL_CMD_CHARS] + f"...[+{len(command) - _MAX_SHELL_CMD_CHARS} chars]"
                shell_commands.append(command)
        return {"tool_counts": tool_counts, "shell_commands": shell_commands, "failed_tool_calls": failed}

    @staticmethod
    def _trace_tail(lines: Sequence[str]) -> list[Any]:
        tail: list[Any] = []
        for line in list(lines)[-_TRACE_TAIL_EVENTS:]:
            try:
                tail.append(json.loads(line))
            except json.JSONDecodeError:
                tail.append(line)
        return tail

    @staticmethod
    def _raise_for_status(returncode: int, result_event: dict[str, Any] | None, stderr_tail: str) -> None:
        if result_event is not None and result_event.get("status") == "success" and returncode == 0:
            return
        if result_event is None:
            message = stderr_tail or f"no terminal 'result' event (exit={returncode})"
        else:
            error = result_event.get("error")
            message = str(error.get("message") if isinstance(error, dict) else error or f"exit={returncode}")
        lowered = message.lower()
        if any(keyword in lowered for keyword in _RATE_LIMIT_KEYWORDS):
            raise ProviderRateLimitError(f"gemini_cli_extract rate-limited: {message}")
        if any(keyword in lowered for keyword in _TRANSIENT_KEYWORDS):
            raise ProviderTransientError(f"gemini_cli_extract transient CLI error: {message}")
        raise ProviderPermanentError(f"gemini_cli_extract CLI error (exit={returncode}): {message}")

    @staticmethod
    def _read_output_json(workdir: Path) -> dict[str, Any]:
        output_path = workdir / "output.json"
        if not output_path.exists():
            raise ProviderPermanentError("gemini_cli_extract: CLI completed but ./output.json was not written.")
        try:
            data = json.loads(output_path.read_text())
        except json.JSONDecodeError as e:
            raise ProviderPermanentError(f"gemini_cli_extract: ./output.json is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ProviderPermanentError(
                f"gemini_cli_extract: ./output.json must be a JSON object, got {type(data).__name__}."
            )
        return data

    def _config_snapshot(self) -> dict[str, Any]:
        return {
            "model": self._model,
            "base_url": self._base_url,
            "thinking_level": self._thinking_level,
            "max_cost_usd": self._max_cost_usd,
            "promote_repeated_structure": self._promote_repeated,
            "additional_properties_false": self._additional_properties_false,
            "evidence_mode": self._evidence_mode,
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.artifact_dir is None:
            raise ProviderConfigError("Gemini CLI requires an artifact directory; use the benchmark runner")
        trace = GeminiCliTrace(
            Path(request.artifact_dir),
            {
                "example_id": request.example_id,
                "pipeline": pipeline.pipeline_name,
                "config": self._config_snapshot(),
                "source_file": request.source_file_path,
            },
        )
        with self._procs_lock:
            self._traces[request.example_id] = trace
        try:
            result = self._run_inference(pipeline, request, trace)
            trace.finish("success")
            return result
        except Exception as exc:
            trace.finish("cancelled" if trace.cancelled.is_set() else "error", error=str(exc))
            if isinstance(exc, ProviderError):
                exc.debug_payload = {**(exc.debug_payload or {}), "trace_path": str(trace.path)}
            raise
        finally:
            with self._procs_lock:
                self._traces.pop(request.example_id, None)

    def _run_inference(
        self, pipeline: PipelineSpec, request: InferenceRequest, trace: GeminiCliTrace
    ) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"{type(self).__name__} only supports EXTRACT product type, got {request.product_type}"
            )
        if request.schema_override is None or len(request.schema_override) == 0:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override."
            )
        src = Path(request.source_file_path)
        if not src.exists():
            raise ProviderPermanentError(f"File not found: {src}")
        ext = src.suffix.lower()
        if ext != ".pdf" and ext not in IMAGE_EXTENSIONS:
            raise ProviderPermanentError(
                f"{type(self).__name__} supports PDFs and {set(IMAGE_EXTENSIONS)}, got {src.suffix}"
            )

        started_at = datetime.now()
        schema = self._prepare_schema(request.schema_override)
        num_pages = page_count(src)

        # The CLI home sits beside the workdir, not inside it, so the agent never sees it.
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            home = Path(tmp) / "home"
            (home / ".gemini").mkdir(parents=True)
            workdir.mkdir()
            (home / ".gemini" / "settings.json").write_text(json.dumps(self._build_settings()), encoding="utf-8")
            staged_name = f"input{ext}"
            shutil.copy2(src, workdir / staged_name)

            prompt = self._build_prompt(schema, staged_name)
            cmd = self._build_cmd()
            lines: deque[str] = deque(maxlen=_TRACE_TAIL_EVENTS)
            tool_usage = {"tool_counts": {}, "shell_commands": [], "failed_tool_calls": 0}
            returncode: int | None = None
            stderr_tail = ""
            agent_citations: Any = None
            request_retry_observed = False
            with GeminiCliTransport(
                self._base_url,
                trace,
                cost_from_request_usage,
                lambda: self.cancel(request.example_id),
                api_key=self._api_key,
                headers=self._headers,
            ) as transport:
                try:
                    returncode, stderr_tail = self._run_cli(
                        cmd,
                        workdir,
                        home,
                        request.example_id,
                        lines,
                        prompt,
                        base_url=transport.cli_base_url,
                        trace=trace,
                        tool_usage=tool_usage,
                    )
                    transport.raise_if_failed()
                    with (home / "gemini_stderr.log").open(encoding="utf-8", errors="replace") as stderr:
                        request_retry_observed = any(_REQUEST_RETRY_PATTERN.search(line) is not None for line in stderr)
                    events = self._events(lines)
                    result_event = next((e for e in reversed(events) if e.get("type") == "result"), None)
                    self._raise_for_status(returncode, result_event, stderr_tail)
                    data = self._read_output_json(workdir)
                    if self._evidence_mode:
                        agent_citations = read_agent_citations_file(workdir / "citations.json")
                except ProviderError as exc:
                    payload: dict[str, Any] = {
                        "returncode": returncode,
                        "stderr_tail": stderr_tail,
                        "trace_tail": self._trace_tail(lines),
                        "config": self._config_snapshot(),
                        "request_retry_observed": request_retry_observed,
                        "request_accounting": transport.summary(),
                    }
                    if isinstance(exc.debug_payload, dict):
                        payload.update(exc.debug_payload)
                    exc.debug_payload = payload
                    raise

        transport.raise_if_failed()
        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        assert result_event is not None
        stats = result_event.get("stats") or {}
        accounting = transport.summary()
        cost_usd = accounting["known_cost_usd"]
        raw_output: dict[str, Any] = {
            "data": data,
            "model": self._model,
            "usage": {"requests": accounting["requests"], "cli_stats": stats},
            "num_pages": num_pages,
            "cost_usd": cost_usd,
            "_cost_complete": accounting["cost_complete"],
            "request_retry_observed": request_retry_observed,
            "cost_per_page_usd": cost_usd / num_pages if cost_usd is not None and num_pages > 0 else None,
            "trace_path": str(trace.path),
            "cost_exceeded_budget": self._max_cost_usd is not None
            and cost_usd is not None
            and cost_usd > self._max_cost_usd,
            "_config": self._config_snapshot(),
            **tool_usage,
            "_trace_tail": self._trace_tail(lines),
        }
        if self._evidence_mode:
            citations, evidence_stats = citations_from_agent_file(agent_citations, source="gemini_cli_evidence")
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
        with self._procs_lock:
            proc = self._procs.get(example_id)
            trace = self._traces.get(example_id)
        if trace is not None:
            trace.cancel("runner cancellation")
        if proc is None:
            return False
        self._terminate(proc)
        return True
