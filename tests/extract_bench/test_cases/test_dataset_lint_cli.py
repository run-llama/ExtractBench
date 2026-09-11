"""Tests for ``extract-bench dataset lint`` exit codes and output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from extract_bench.test_cases.cli import DatasetCLI


def _extract_sidecar(extra_key: bool = False) -> dict[str, Any]:
    output: dict[str, Any] = {"po_number": "PO-1"}
    if extra_key:
        output["vendor"] = "Acme"
    return {
        "data_schema": {"type": "object", "properties": {"po_number": {"type": "string"}}},
        "expected_output": output,
        "_field_rules": {"po_number": {"evidence": [{"page": 1, "value": "PO-1"}]}},
    }


def _write(root: Path, name: str, payload: dict[str, Any]) -> None:
    (root / name).write_text(json.dumps(payload), encoding="utf-8")


def test_clean_tree_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path, "good.test.json", _extract_sidecar())
    assert DatasetCLI().lint(str(tmp_path)) == 0
    assert "1 extract sidecar(s) linted, 0 with findings" in capsys.readouterr().out


def test_findings_exit_one_and_name_the_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "sub").mkdir()
    _write(tmp_path, "good.test.json", _extract_sidecar())
    _write(tmp_path / "sub", "bad.test.json", _extract_sidecar(extra_key=True))
    _write(tmp_path, "parse.test.json", {"test_rules": [{"type": "present", "text": "x"}]})
    assert DatasetCLI().lint(str(tmp_path), format="json") == 1
    report = json.loads(capsys.readouterr().out)
    assert (report["linted"], report["skipped"], report["files_with_findings"]) == (2, 1, 1)
    assert report["findings"] == [
        {
            "file": "sub/bad.test.json",
            "code": "output.unknown_field",
            "path": "expected_output/vendor",
            "message": "key is not declared by data_schema",
        }
    ]


def test_text_format_prints_one_error_line_per_finding(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path, "bad.test.json", _extract_sidecar(extra_key=True))
    assert DatasetCLI().lint(str(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "ERROR bad.test.json: output.unknown_field expected_output/vendor: key is not declared" in out


def test_legacy_rules_pass_only_when_allowed(tmp_path: Path) -> None:
    payload = _extract_sidecar()
    payload["test_rules"] = [
        {"type": "extract_field", "field_path": "po_number", **payload.pop("_field_rules")["po_number"]}
    ]
    _write(tmp_path, "legacy.test.json", payload)
    assert DatasetCLI().lint(str(tmp_path)) == 1
    assert DatasetCLI().lint(str(tmp_path), allow_legacy_rules=True) == 0


def test_unreadable_sidecar_is_a_finding(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "broken.test.json").write_text("{not json", encoding="utf-8")
    assert DatasetCLI().lint(str(tmp_path), format="json") == 1
    assert json.loads(capsys.readouterr().out)["findings"][0]["code"] == "file.unreadable"


def test_tree_without_extract_sidecars_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path, "parse.test.json", {"test_rules": []})
    assert DatasetCLI().lint(str(tmp_path)) == 2
    assert "no extract sidecars" in capsys.readouterr().err


def test_bad_arguments_exit_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert DatasetCLI().lint(str(tmp_path / "missing")) == 1
    assert DatasetCLI().lint(str(tmp_path), format="yaml") == 1
    assert "unknown format" in capsys.readouterr().err
