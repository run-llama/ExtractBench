from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required")

from extract_bench.inference.providers.extract import claude_code_extract
from extract_bench.inference.providers.extract.claude_code_extract import ClaudeCodeExtractProvider
from extract_bench.schemas.pipeline_io import RawInferenceResult


def _large_schema() -> dict[str, Any]:
    fields = {f"field_{i:04d}": {"type": "string", "description": "x" * 100} for i in range(1500)}
    return {"type": "object", "properties": fields}


def test_small_schema_stays_inline(tmp_path: Path) -> None:
    provider = ClaudeCodeExtractProvider("claude_code_extract")
    schema = provider._prepare_schema({"type": "object", "properties": {"invoice_number": {"type": "string"}}})
    prompt, delivery = provider._stage_prompt(schema, "input.pdf", tmp_path)
    assert delivery == "inline"
    assert prompt == provider._build_prompt(schema, "input.pdf")
    assert json.dumps(schema, indent=2) in prompt
    assert not (tmp_path / "schema.json").exists()


def test_oversized_schema_moves_to_file(tmp_path: Path) -> None:
    provider = ClaudeCodeExtractProvider("claude_code_extract", {"evidence_mode": True})
    schema = provider._prepare_schema(_large_schema())
    assert len(provider._build_prompt(schema, "input.pdf").encode()) > claude_code_extract._MAX_INLINE_PROMPT_BYTES

    prompt, delivery = provider._stage_prompt(schema, "input.pdf", tmp_path)
    assert delivery == "file"
    assert json.loads((tmp_path / "schema.json").read_text()) == schema
    assert "`./schema.json`" in prompt
    assert "field_0000" not in prompt
    assert len(prompt.encode()) < claude_code_extract._MAX_INLINE_PROMPT_BYTES
    # The rest of the ask, evidence rule included, is unchanged.
    assert prompt.endswith(provider._build_prompt(schema, "input.pdf").split("```\n\n", 1)[1])


@pytest.fixture
def fake_claude(tmp_path: Path) -> Path:
    """A local CLI double; invokes no model and fails if updates are enabled."""
    import sys

    executable = tmp_path / "claude"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "assert os.environ['DISABLE_AUTOUPDATER'] == '1'\n"
        "assert os.environ['DISABLE_UPDATES'] == '1'\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('2.1.280 (Claude Code)')\n"
        "else:\n"
        "    Path(__file__).with_suffix('.invoked').touch()\n"
        "    Path('output.json').write_text('{}')\n"
        "    print(json.dumps({'type': 'result', 'is_error': False, 'total_cost_usd': 0}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _run_with_fake_cli(provider: ClaudeCodeExtractProvider, tmp_path: Path) -> RawInferenceResult:
    from extract_bench.schemas.pipeline import PipelineSpec
    from extract_bench.schemas.pipeline_io import InferenceRequest

    source = tmp_path / "input.png"
    source.write_bytes(b"image")
    pipeline = PipelineSpec.model_validate(
        {"pipeline_name": "claude_test", "provider_name": "claude_code_extract", "product_type": "extract"},
    )
    request = InferenceRequest.model_validate(
        {
            "example_id": "invoice",
            "source_file_path": str(source),
            "product_type": "extract",
            "schema_override": {"type": "object"},
            "artifact_dir": str(tmp_path / "traces"),
        },
    )
    return provider.run_inference(pipeline, request)


@pytest.mark.parametrize("pinned", [True, False])
def test_cli_records_actual_version_and_disables_updates(
    fake_claude: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pinned: bool,
) -> None:
    monkeypatch.delenv("CLAUDE_CODE_VERSION", raising=False)
    monkeypatch.setenv("DISABLE_AUTOUPDATER", "0")
    monkeypatch.setenv("DISABLE_UPDATES", "0")
    config = {"claude_bin": str(fake_claude)}
    if pinned:
        config["claude_version"] = "2.1.280"
    provider = ClaudeCodeExtractProvider("claude_code_extract", config)
    raw = _run_with_fake_cli(provider, tmp_path)
    assert raw.raw_output["harness"] == {
        "executable": str(fake_claude.resolve()),
        "version": "2.1.280",
        "requested_version": "2.1.280" if pinned else None,
    }
    assert fake_claude.with_suffix(".invoked").exists()


def test_version_mismatch_prevents_extraction(
    fake_claude: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extract_bench.inference.providers.base import ProviderConfigError

    monkeypatch.setenv("DISABLE_AUTOUPDATER", "1")
    monkeypatch.setenv("DISABLE_UPDATES", "1")
    provider = ClaudeCodeExtractProvider(
        "claude_code_extract",
        {"claude_bin": str(fake_claude), "claude_version": "2.1.279"},
    )
    with pytest.raises(ProviderConfigError, match="version mismatch") as caught:
        _run_with_fake_cli(provider, tmp_path)
    assert caught.value.debug_payload["harness"]["version"] == "2.1.280"
    assert not fake_claude.with_suffix(".invoked").exists()


def test_timeout_keeps_version_record_from_before_extraction(
    fake_claude: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_VERSION", "2.1.280")
    provider = ClaudeCodeExtractProvider("claude_code_extract", {"claude_bin": str(fake_claude)})

    def timeout(*args: object) -> None:
        records = list((tmp_path / "traces" / "claude_code").glob("*.harness.json"))
        record = json.loads(records[0].read_text(encoding="utf-8"))
        assert record["example_id"] == "invoice"
        assert record["version"] == record["requested_version"] == "2.1.280"
        raise TimeoutError("document deadline")

    monkeypatch.setattr(provider, "_run_cli", timeout)
    with pytest.raises(TimeoutError, match="document deadline"):
        _run_with_fake_cli(provider, tmp_path)
    assert len(list((tmp_path / "traces" / "claude_code").glob("*.harness.json"))) == 1


@pytest.mark.parametrize("version", ["latest", "^2.1.280", "2.1", "", 2.1])
def test_nonexact_pins_are_rejected(version: object) -> None:
    from extract_bench.inference.providers.base import ProviderConfigError

    with pytest.raises(ProviderConfigError, match="exact version"):
        ClaudeCodeExtractProvider("claude_code_extract", {"claude_version": version})


def test_cancellation_during_version_probe_prevents_extraction(
    fake_claude: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Mapping

    from extract_bench.inference.limits import ExecutionBudget

    provider = ClaudeCodeExtractProvider("claude_code_extract", {"claude_bin": str(fake_claude)})
    budget = ExecutionBudget()
    resolve = provider._resolve_harness

    def cancel_after_probe() -> Mapping[str, str | None]:
        harness = resolve()
        budget.cancel()
        return harness

    monkeypatch.setattr(provider, "_resolve_harness", cancel_after_probe)
    with budget.activate(), pytest.raises(TimeoutError, match="deadline"):
        _run_with_fake_cli(provider, tmp_path)
    assert not fake_claude.with_suffix(".invoked").exists()
