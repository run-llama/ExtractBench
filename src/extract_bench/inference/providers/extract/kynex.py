"""Run a pinned KYNEX harness in an automatically provisioned virtual environment.

The first extraction clones https://github.com/Nikit117/kynex-harness and
installs its requirements into an isolated cache. Later extractions reuse that
installation. Set KYNEX_CACHE_DIR to choose the cache location; Azure OpenAI
and Document Intelligence credentials come from the process environment.

The upstream agent and post-processing rules run unchanged. No per-field
citation boxes are emitted, so field_citations is empty.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

logger = logging.getLogger(__name__)

HARNESS_REPOSITORY = "https://github.com/Nikit117/kynex-harness.git"
HARNESS_REVISION = "d50a78d9ee99a076e05e8d8942dfe3e2152e9065"
_SETUP_TIMEOUT_S = 900

# Preserve the rule switches from the submitted benchmark configuration.
HARNESS_ENV: dict[str, str] = {
    "ADH_BULK_ARRAYS": "1",
    "ADH_PRUNE_NON_RECORDS": "1",
    "ADH_REDACTION_MARKERS": "1",
    "ADH_OCR_FALLBACK": "1",
    "ADH_BULK_OCR": "1",
    "ADH_STRUCTURED_ARRAYS": "1",
}


def _run_setup(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=_SETUP_TIMEOUT_S,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr if exc.stderr is not None and len(exc.stderr) > 0 else exc.stdout
        if detail is None:
            detail = str(exc)
        raise ProviderConfigError(f"KYNEX setup failed: {detail[-2000:]}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProviderConfigError(f"KYNEX setup failed: {exc}") from exc
    return result.stdout.strip()


def _ensure_harness() -> tuple[Path, Path]:
    """Publish a complete pinned installation under a process-shared lock."""
    root = Path(os.environ.get("KYNEX_CACHE_DIR", str(Path.home() / ".cache" / "extract-bench" / "kynex")))
    cache = root.expanduser().resolve() / HARNESS_REVISION / f"py{sys.version_info.major}.{sys.version_info.minor}"
    cache.mkdir(parents=True, exist_ok=True)
    repo = cache / "repo"
    environment = cache / "venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    ready = cache / "ready"
    try:
        with FileLock(cache / "setup.lock", timeout=_SETUP_TIMEOUT_S):
            if not repo.exists():
                logger.info("Cloning KYNEX at %s", HARNESS_REVISION)
                with tempfile.TemporaryDirectory(prefix="clone-", dir=cache) as staging:
                    checkout = Path(staging) / "repo"
                    _run_setup(
                        [
                            "git",
                            "clone",
                            "--no-checkout",
                            HARNESS_REPOSITORY,
                            str(checkout),
                        ]
                    )
                    _run_setup(["git", "checkout", "--detach", HARNESS_REVISION], cwd=checkout)
                    checkout.rename(repo)
            actual_revision = _run_setup(["git", "rev-parse", "HEAD"], cwd=repo)
            changes = _run_setup(["git", "status", "--porcelain"], cwd=repo)
            if actual_revision != HARNESS_REVISION or len(changes) > 0:
                raise ProviderConfigError(f"Cached KYNEX checkout differs from {HARNESS_REVISION}: {repo}")
            if ready.exists() and python.exists() and ready.read_text(encoding="utf-8") == HARNESS_REVISION:
                return repo, python
            ready.unlink(missing_ok=True)
            logger.info("Installing KYNEX dependencies in %s", environment)
            _run_setup([sys.executable, "-m", "venv", str(environment)])
            _run_setup(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-input",
                    "--requirement",
                    str(repo / "requirements.txt"),
                ],
                cwd=repo,
            )
            ready.write_text(HARNESS_REVISION, encoding="utf-8")
    except Timeout as exc:
        raise ProviderConfigError(f"Timed out waiting for KYNEX setup: {cache}") from exc
    return repo, python


async def _extract_async(
    source: Path, schema: dict[str, Any], timeout: float, repo: Path, python: Path
) -> dict[str, Any]:
    """Run one extraction in the managed harness environment."""

    def _run() -> dict[str, Any]:
        script = repo / "autonomous_document_harness" / "scripts" / "extract_one.py"
        with tempfile.TemporaryDirectory() as td:
            schema_file = Path(td) / "schema.json"
            schema_file.write_text(json.dumps(schema), encoding="utf-8")
            out_file = Path(td) / "extraction.json"
            cmd = [
                str(python),
                "-I",
                str(script),
                "--source",
                str(source),
                "--schema",
                str(schema_file),
                "--out",
                str(out_file),
            ]
            env = os.environ.copy()
            env.update(HARNESS_ENV)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(repo),
                timeout=timeout,
                env=env,
            )
            if proc.returncode != 0:
                raise ProviderPermanentError(
                    f"harness extraction failed (exit {proc.returncode}): {(proc.stderr or proc.stdout)[-400:]}"
                )
            if not out_file.exists():
                raise ProviderPermanentError("harness produced no extraction file")
            extraction = json.loads(out_file.read_text(encoding="utf-8"))
        return {"extraction": extraction, "harness_revision": HARNESS_REVISION}

    return await asyncio.get_running_loop().run_in_executor(None, _run)


@register_provider("kynex_extract")
class KynexExtractProvider(Provider):
    """KYNEX extraction with a per-document timeout (default: 3600 seconds)."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._timeout = int(self.base_config.get("timeout", 3600))

    def run_inference(self, pipeline: Any, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"KynexExtractProvider only supports EXTRACT product type, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError("schema_override is required for EXTRACT product type.")
        source = Path(request.source_file_path).resolve()
        if not source.exists():
            raise ProviderPermanentError(f"File not found: {source}")

        repo, python = _ensure_harness()
        started_at = datetime.now()
        try:
            raw_output = asyncio.run(
                asyncio.wait_for(
                    _extract_async(source, request.schema_override, self._timeout, repo, python),
                    timeout=self._timeout,
                )
            )
        except TimeoutError as exc:
            raise ProviderPermanentError(f"harness extraction timed out after {self._timeout}s") from exc
        except ProviderPermanentError:
            raise
        except Exception as exc:  # noqa: BLE001 -- report as a document failure
            raise ProviderPermanentError(f"harness extraction failed: {exc}") from exc

        completed_at = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        extraction = raw_result.raw_output.get("extraction")
        extracted_data = extraction if isinstance(extraction, (dict, list)) else {}
        output = ExtractOutput(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data,
            field_citations=[],
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
