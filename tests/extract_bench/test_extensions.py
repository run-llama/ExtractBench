"""The public extension surface and package metadata stay importable and consistent."""

from __future__ import annotations

import re
from pathlib import Path

import extract_bench
from extract_bench import extensions
from extract_bench.cli import BenchCLI


def test_extensions_exports_registration_hooks() -> None:
    for name in extensions.__all__:
        assert callable(getattr(extensions, name)), name
    assert {"register_provider", "register_pipeline"} <= set(extensions.__all__)


def test_version_is_the_single_source_of_truth() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", extract_bench.__version__)
    assert BenchCLI().version() == extract_bench.__version__
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in pyproject
    assert 'path = "src/extract_bench/__init__.py"' in pyproject


def test_package_is_typed() -> None:
    assert (Path(extract_bench.__file__).parent / "py.typed").exists()
