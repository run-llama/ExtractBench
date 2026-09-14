"""Discovery keeps one input per stem and ignores per-doc artifact directories."""

from __future__ import annotations

import json
from pathlib import Path

from extract_bench.test_cases.loader import _dedupe_stem_twins, _is_artifact_dir, load_test_cases

_SIDECAR = {
    "data_schema": {"type": "object", "properties": {"total": {"type": "number"}}},
    "expected_output": {"total": 1},
}


def _write_case(directory: Path, stem: str, suffixes: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        (directory / f"{stem}{suffix}").write_bytes(b"%PDF-1.4\n")
    (directory / f"{stem}.test.json").write_text(json.dumps(_SIDECAR))


def test_dedupe_prefers_document_over_image_twin(capsys) -> None:
    files = [Path("g/doc.png"), Path("g/doc.pdf"), Path("g/other.png")]
    kept = _dedupe_stem_twins(files)
    assert kept == [Path("g/doc.pdf"), Path("g/other.png")]
    assert "share stem 'doc'" in capsys.readouterr().out


def test_dedupe_keeps_same_stem_in_different_dirs() -> None:
    files = [Path("a/doc.pdf"), Path("b/doc.pdf")]
    assert _dedupe_stem_twins(files) == files


def test_is_artifact_dir() -> None:
    assert _is_artifact_dir(Path("x/doc.pdf.images"))
    assert _is_artifact_dir(Path("x/doc.parse"))
    assert _is_artifact_dir(Path("x/doc.v2.screenshots"))
    assert not _is_artifact_dir(Path("x/invoices"))


def test_grouped_dataset_loads_one_case_per_stem(tmp_path: Path) -> None:
    _write_case(tmp_path / "invoices", "doc", (".pdf", ".png"))
    cases = load_test_cases(tmp_path, product_type="EXTRACT")
    assert [case.test_id for case in cases] == ["invoices/doc"]
    assert cases[0].file_path.suffix == ".pdf"


def test_flat_dataset_ignores_artifact_dir(tmp_path: Path) -> None:
    _write_case(tmp_path, "doc", (".pdf",))
    (tmp_path / "doc.pdf.images").mkdir()
    (tmp_path / "doc.pdf.images" / "crop.png").write_bytes(b"")
    cases = load_test_cases(tmp_path, product_type="EXTRACT")
    assert [case.test_id for case in cases] == [f"{tmp_path.name}/doc"]
