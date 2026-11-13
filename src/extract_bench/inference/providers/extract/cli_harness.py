"""Exact CLI version checks and durable provenance for coding-agent attempts."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extract_bench.inference.limits import execution_budget
from extract_bench.inference.providers.base import ProviderConfigError
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest

HARNESS_VERSION = "harness_version"
CLAUDE_UPDATE_ENV = {"DISABLE_AUTOUPDATER": "1", "DISABLE_UPDATES": "1"}
_EXACT_VERSION = r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?"
_VERSION_OUTPUT = re.compile(rf"(?:codex-cli )?({_EXACT_VERSION})(?: \(Claude Code\))?")
# Local startup only; inference keeps the runner's full execution budget.
_VERSION_PROBE_TIMEOUT_S = 30


@dataclass
class _Probe:
    cancelled: threading.Event = field(default_factory=threading.Event)
    process: subprocess.Popen[str] | None = None


class CliHarness:
    def __init__(
        self,
        name: str,
        executable: str,
        version: str | None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if version is not None and (not isinstance(version, str) or re.fullmatch(_EXACT_VERSION, version) is None):
            raise ProviderConfigError(f"{HARNESS_VERSION} must be an exact version, such as 2.1.280")
        self.name = name
        self.executable = executable
        self.version = version
        self.env = dict(env) if env is not None else {}
        self._probes: dict[str, _Probe] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _persist(path: Path | None, record: Mapping[str, Any]) -> None:
        if path is None:
            return
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    def prepare(self, pipeline: PipelineSpec, request: InferenceRequest) -> Mapping[str, Any]:
        # Persist intent before probing: even a hung --version has an attempt
        # record. UUIDs preserve separate retries and concurrent documents.
        record: dict[str, Any] = {
            "example_id": request.example_id,
            "pipeline_name": pipeline.pipeline_name,
            "recorded_at": datetime.now(UTC).isoformat(),
            "name": self.name,
            "executable": self.executable,
            "version": None,
            "requested_version": self.version,
        }
        path = None
        if request.artifact_dir is not None:
            directory = Path(request.artifact_dir) / self.name
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{uuid.uuid4().hex}.harness.json"
        state = _Probe()
        with self._lock:
            self._probes[request.example_id] = state
        try:
            self._persist(path, record)
            executable = shutil.which(self.executable)
            if executable is None:
                raise ProviderConfigError(f"{self.name} executable not found: {self.executable}")
            # Use the native-install symlink target for both probe and extraction.
            record["executable"] = str(Path(executable).resolve())
            self._persist(path, record)
            budget = execution_budget()
            with self._lock:
                if state.cancelled.is_set():
                    raise TimeoutError(f"{self.name} version probe cancelled")
                timeout = min(_VERSION_PROBE_TIMEOUT_S, budget.remaining())
                state.process = subprocess.Popen(
                    [record["executable"], "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={**os.environ, **self.env},
                )
            try:
                stdout, stderr = state.process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(f"{self.name} version probe timed out after {timeout:.1f}s") from error
            if state.cancelled.is_set():
                raise TimeoutError(f"{self.name} version probe cancelled")
            budget.remaining()
            if state.process.returncode != 0:
                raise ProviderConfigError(f"{self.name} version probe failed: {stderr.strip()}")
            match = _VERSION_OUTPUT.fullmatch(stdout.strip())
            if match is None:
                raise ProviderConfigError(f"Unrecognized {self.name} version output: {stdout.strip()!r}")
            record["version"] = match[1]
            if self.version is not None and match[1] != self.version:
                raise ProviderConfigError(
                    f"{self.name} version mismatch: requested {self.version}, found {match[1]}. "
                    "Install the requested version or select its executable in pipeline config.",
                    debug_payload={"harness": dict(record)},
                )
            self._persist(path, record)
            return record
        except Exception as error:
            record["error"] = str(error)
            self._persist(path, record)
            raise
        finally:
            with self._lock:
                self._probes.pop(request.example_id, None)
            if state.process is not None:
                if state.process.poll() is None:
                    with suppress(ProcessLookupError):
                        state.process.kill()
                state.process.communicate()

    def cancel(self, example_id: str) -> bool:
        with self._lock:
            state = self._probes.get(example_id)
            if state is None:
                return False
            state.cancelled.set()
            if state.process is not None and state.process.poll() is None:
                with suppress(ProcessLookupError):
                    state.process.kill()
        return True
