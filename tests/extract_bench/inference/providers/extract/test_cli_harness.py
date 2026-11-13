"""Exercise CLI pins through real local processes, without model calls."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

pytest.importorskip("pypdf", reason="runners extra required")

from extract_bench.inference.providers.base import ProviderConfigError
from extract_bench.inference.providers.extract import cli_harness
from extract_bench.inference.providers.extract.claude_code_extract import ClaudeCodeExtractProvider
from extract_bench.inference.providers.extract.codex_code_extract import CodexCodeExtractProvider
from extract_bench.inference.providers.extract.codex_deepseek_extract import (
    CodexDeepSeekExtractProvider,
    CodexGLMExtractProvider,
)
from extract_bench.inference.providers.extract.gemini_cli_extract import GeminiCliExtractProvider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest


@pytest.fixture(
    params=[
        (ClaudeCodeExtractProvider, "claude_bin", "2.1.280 (Claude Code)"),
        (CodexCodeExtractProvider, "codex_bin", "codex-cli 0.123.0"),
        (CodexDeepSeekExtractProvider, "codex_bin", "codex-cli 0.123.0"),
        (CodexGLMExtractProvider, "codex_bin", "codex-cli 0.123.0"),
        (GeminiCliExtractProvider, "gemini_bin", "0.59.0"),
    ]
)
def attempt(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, PipelineSpec, InferenceRequest, Path]:
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    provider_type, binary_key, version_output = request.param
    version = version_output.split()[1 if binary_key == "codex_bin" else 0]
    executable = tmp_path / "cli-target"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        f"kind = {binary_key!r}\n"
        "if kind == 'claude_bin':\n"
        "    assert os.environ['DISABLE_AUTOUPDATER'] == '1'\n"
        "    assert os.environ['DISABLE_UPDATES'] == '1'\n"
        "if sys.argv[1:] == ['--version']:\n"
        f"    print({version_output!r})\n"
        "else:\n"
        "    if kind == 'codex_bin':\n"
        "        assert 'check_for_update_on_startup=false' in sys.argv\n"
        "    if kind == 'gemini_bin':\n"
        "        settings = json.loads((Path(os.environ['GEMINI_CLI_HOME']) / '.gemini/settings.json').read_text())\n"
        "        assert settings['general']['enableAutoUpdate'] is False\n"
        "    if kind != 'claude_bin': sys.stdin.read()\n"
        "    Path(__file__).with_suffix('.invoked').touch()\n"
        "    Path('output.json').write_text('{}')\n"
        "    print(json.dumps({'type':'item.completed','item':{'type':'command_execution',"
        " 'command':'extract', 'exit_code':0,'status':'completed'}}))\n"
        "    print(json.dumps({'type':'turn.completed','usage':{}}))\n"
        "    print(json.dumps({'type':'result','status':'success','is_error':False,'total_cost_usd':0,'stats':{}}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    launcher = tmp_path / "cli"
    launcher.symlink_to(executable)
    config = {binary_key: str(launcher), "harness_version": version, "api_key": "test"}
    provider = provider_type("test", config)
    pipeline = PipelineSpec.model_validate(
        {
            "pipeline_name": "test",
            "provider_name": "test",
            "product_type": "extract",
            "config": config,
        }
    )
    source = tmp_path / "input.png"
    source.write_bytes(b"image")
    inference_request = InferenceRequest.model_validate(
        {
            "example_id": "invoice",
            "source_file_path": str(source),
            "product_type": "extract",
            "schema_override": {"type": "object"},
            "artifact_dir": str(tmp_path / "traces"),
        }
    )
    return provider, pipeline, inference_request, executable


def test_selected_version_reaches_output_and_survives_launcher_change(
    attempt: tuple[Any, PipelineSpec, InferenceRequest, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, pipeline, request, target = attempt
    prepare = provider._harness.prepare

    def swing_launcher(*args: Any) -> Any:
        record = prepare(*args)
        launcher = target.parent / "cli"
        launcher.unlink()
        launcher.symlink_to(target.parent / "nonexistent-new-version")
        return record

    monkeypatch.setattr(provider._harness, "prepare", swing_launcher)
    raw = provider.run_inference(pipeline, request)
    assert raw.raw_output["harness"]["version"] == pipeline.config["harness_version"]
    assert raw.raw_output["harness"]["executable"] == str(target.resolve())
    assert raw.raw_output["_config"]["harness_version"] == pipeline.config["harness_version"]
    assert target.with_suffix(".invoked").exists()


def test_mismatch_stops_extraction_and_preserves_observed_version(
    attempt: tuple[Any, PipelineSpec, InferenceRequest, Path],
) -> None:
    provider, pipeline, request, target = attempt
    provider._harness.version = "0.0.1"
    with pytest.raises(ProviderConfigError, match="version mismatch"):
        provider.run_inference(pipeline, request)
    record = json.loads(next(Path(request.artifact_dir).rglob("*.harness.json")).read_text(encoding="utf-8"))
    assert record["version"] == pipeline.config["harness_version"]
    assert record["requested_version"] == "0.0.1"
    assert not target.with_suffix(".invoked").exists()


def test_cancel_kills_active_probe_and_preserves_failed_attempt(
    attempt: tuple[Any, PipelineSpec, InferenceRequest, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, pipeline, request, target = attempt
    target.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n", encoding="utf-8")
    started = threading.Event()
    popen = cli_harness.subprocess.Popen
    processes = []

    def start(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        process = popen(*args, **kwargs)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(cli_harness.subprocess, "Popen", start)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(provider.run_inference, pipeline, request)
        try:
            assert started.wait(5), "version probe did not start"
            assert provider.cancel(request.example_id)
            with pytest.raises(TimeoutError, match="probe cancelled"):
                future.result(timeout=5)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
    record = json.loads(next(Path(request.artifact_dir).rglob("*.harness.json")).read_text(encoding="utf-8"))
    assert record["version"] is None
    assert record["requested_version"] == pipeline.config["harness_version"]
    assert "cancelled" in record["error"]
    assert not target.with_suffix(".invoked").exists()


@pytest.mark.parametrize("version", ["latest", "^2.1.280", "2.1", "", 2.1])
def test_nonexact_pins_are_rejected(version: Any) -> None:
    with pytest.raises(ProviderConfigError, match="exact version"):
        cli_harness.CliHarness("test", "test", version)


@pytest.mark.parametrize("single_file", [True, False])
def test_cli_pin_overrides_registry_config_without_mutating_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    single_file: bool,
) -> None:
    from extract_bench.cli import BenchCLI
    from extract_bench.inference import cli as inference_cli
    from extract_bench.test_cases.schema import ExtractTestCase

    source = tmp_path / "doc.png"
    source.write_bytes(b"image")
    source.with_suffix(".test.json").write_text('{"schema":{"type":"object"}}', encoding="utf-8")
    registered = PipelineSpec.model_validate(
        {
            "pipeline_name": "pinned_cli",
            "provider_name": "claude_code_extract",
            "product_type": "extract",
            "config": {"model": "claude-opus-4-8", "harness_version": "2.1.279"},
        }
    )
    case = ExtractTestCase.model_validate(
        {
            "test_id": "doc",
            "group": "test",
            "file_path": source,
            "schema": {"type": "object"},
        }
    )
    captured = []

    def capture(spec: PipelineSpec) -> None:
        captured.append(spec)
        raise ProviderConfigError("stop before inference")

    monkeypatch.setattr(inference_cli, "get_pipeline", lambda name: registered)
    monkeypatch.setattr(inference_cli, "load_test_cases", lambda **kwargs: [case])
    monkeypatch.setattr(inference_cli, "create_provider", capture)
    inputs = {"file": source} if single_file else {"input_dir": tmp_path}
    assert (
        BenchCLI().run(
            "pinned_cli",
            **inputs,
            output_dir=tmp_path / "output",
            harness_version="2.1.280",
            open_report=False,
        )
        == 1
    )
    assert captured[0].config == {"model": "claude-opus-4-8", "harness_version": "2.1.280"}
    assert registered.config["harness_version"] == "2.1.279"


def test_probe_timeout_is_classified_and_keeps_attempt_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = Mock()
    process.communicate.side_effect = [subprocess.TimeoutExpired("cli", 30), ("", "")]
    process.poll.return_value = None
    monkeypatch.setattr(cli_harness.subprocess, "Popen", lambda *args, **kwargs: process)
    harness = cli_harness.CliHarness("test", sys.executable, "1.2.3")
    pipeline = PipelineSpec.model_validate(
        {"pipeline_name": "test", "provider_name": "test", "product_type": "extract"},
    )
    request = InferenceRequest.model_validate(
        {
            "example_id": "doc",
            "source_file_path": str(tmp_path / "doc.pdf"),
            "product_type": "extract",
            "artifact_dir": str(tmp_path),
        },
    )
    with pytest.raises(TimeoutError, match="probe timed out"):
        harness.prepare(pipeline, request)
    record = json.loads(next(tmp_path.rglob("*.harness.json")).read_text(encoding="utf-8"))
    assert record["version"] is None
    assert record["requested_version"] == "1.2.3"
    assert "timed out" in record["error"]
    process.kill.assert_called_once()
    assert process.communicate.call_count == 2


def test_extraction_timeout_keeps_observed_version(
    attempt: tuple[Any, PipelineSpec, InferenceRequest, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, pipeline, request, _ = attempt

    def timeout(*args: Any, **kwargs: Any) -> None:
        raise TimeoutError("extraction expired")

    monkeypatch.setattr(provider, "_run_cli", timeout)
    with pytest.raises(TimeoutError, match="extraction expired"):
        provider.run_inference(pipeline, request)
    record = json.loads(next(Path(request.artifact_dir).rglob("*.harness.json")).read_text(encoding="utf-8"))
    assert record["version"] == pipeline.config["harness_version"]
    assert record["example_id"] == request.example_id
