from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import pytest

from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.base import ProviderConfigError
from extract_bench.inference.providers.extract import kynex
from extract_bench.inference.providers.registry import create_provider
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """A local harness with a real installable dependency and a newer, wrong HEAD."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Test fixture")
    wheel = repo / "fixture_dependency-1.0-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        files = {
            "fixture_dependency/__init__.py": 'VALUE = "installed dependency"\n',
            "fixture_dependency-1.0.dist-info/METADATA": (
                "Metadata-Version: 2.1\nName: fixture-dependency\nVersion: 1.0\n"
            ),
            "fixture_dependency-1.0.dist-info/WHEEL": (
                "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
            ),
        }
        for name, text in files.items():
            archive.writestr(name, text)
        record = "fixture_dependency-1.0.dist-info/RECORD"
        archive.writestr(record, "".join(f"{name},,\n" for name in [*files, record]))
    (repo / "requirements.txt").write_text(f"./{wheel.name}\n", encoding="utf-8")
    entry = repo / "autonomous_document_harness" / "scripts" / "extract_one.py"
    entry.parent.mkdir(parents=True)
    entry.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "from fixture_dependency import VALUE\n"
        "p = argparse.ArgumentParser()\n"
        "for name in ('source', 'schema', 'out'): p.add_argument('--' + name, required=True)\n"
        "a = p.parse_args()\n"
        "assert json.loads(Path(a.schema).read_text())['type'] == 'object'\n"
        "Path(a.out).write_text(json.dumps({'value': VALUE, 'source': Path(a.source).read_text()}))\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Pinned harness")
    revision = _git(repo, "rev-parse", "HEAD")
    entry.write_text("raise RuntimeError('wrong revision')\n", encoding="utf-8")
    _git(repo, "commit", "-am", "Newer upstream revision must not run")
    monkeypatch.setattr(kynex, "HARNESS_REPOSITORY", str(repo))
    monkeypatch.setattr(kynex, "HARNESS_REVISION", revision)
    monkeypatch.setenv("KYNEX_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    return repo, revision


def test_registered_pipeline_installs_pinned_harness_once_for_concurrent_documents(
    harness: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch missing registration, running upstream HEAD, missing deps, and setup races."""
    pipeline = get_pipeline("kynex_gpt_5_4")
    provider = create_provider(pipeline)
    setup_commands: list[Sequence[str]] = []
    original = kynex._run_setup

    def track(command: Sequence[str], *, cwd: Path | None = None) -> str:
        setup_commands.append(command)
        return original(command, cwd=cwd)

    monkeypatch.setattr(kynex, "_run_setup", track)
    with ThreadPoolExecutor(max_workers=2) as pool:
        installations = list(pool.map(lambda _: kynex._ensure_harness(), range(2)))
    repo, python = installations[0]
    assert installations[1] == (repo, python)
    assert _git(repo, "rev-parse", "HEAD") == harness[1]
    assert python != Path(sys.executable)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "invoice.pdf").write_text("document input", encoding="utf-8")
    request = InferenceRequest(
        example_id="fixture",
        source_file_path="invoice.pdf",
        product_type=ProductType.EXTRACT,
        schema_override={"type": "object", "properties": {"value": {"type": "string"}}},
    )
    raw = provider.run_inference(pipeline, request)
    output = provider.normalize(raw)
    assert output.output.extracted_data == {
        "value": "installed dependency",
        "source": "document input",
    }
    assert raw.raw_output["harness_revision"] == harness[1]
    assert sum("clone" in command for command in setup_commands) == 1
    assert sum("pip" in command for command in setup_commands) == 1

    (repo / "requirements.txt").write_text("unexpected edit\n", encoding="utf-8")
    with pytest.raises(ProviderConfigError, match="differs from"):
        kynex._ensure_harness()


def test_failed_dependency_install_is_retried_instead_of_reused_as_ready(
    harness: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted/failed installation must not publish a reusable environment."""
    original = kynex._run_setup

    def fail_install(command: Sequence[str], *, cwd: Path | None = None) -> str:
        if "pip" in command:
            raise ProviderConfigError("dependency install interrupted")
        return original(command, cwd=cwd)

    monkeypatch.setattr(kynex, "_run_setup", fail_install)
    with pytest.raises(ProviderConfigError, match="dependency install interrupted"):
        kynex._ensure_harness()
    monkeypatch.setattr(kynex, "_run_setup", original)
    repo, python = kynex._ensure_harness()
    assert _git(repo, "rev-parse", "HEAD") == harness[1]
    result = subprocess.check_output(
        [
            str(python),
            "-c",
            "import fixture_dependency; print(fixture_dependency.VALUE)",
        ],
        text=True,
    )
    assert result.strip() == "installed dependency"
