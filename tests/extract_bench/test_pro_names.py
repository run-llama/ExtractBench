import json
from pathlib import Path

import pytest

from extract_bench.data.pro_names import LEGACY_PRO_SPLITS, migrate_pro_layout
from extract_bench.data.pro_release import stage_pro_snapshot


def test_migration_reuses_cached_predictions_without_rewriting_extracted_values(tmp_path: Path):
    """Renaming a result path without its envelope IDs would miss the inference cache."""
    group = tmp_path / "house_vocabulary"
    group.mkdir()
    result = {
        "request": {"example_id": "house_vocabulary/sample", "source_file_path": "/data/house_vocabulary/sample.pdf"},
        "output": {
            "example_id": "house_vocabulary/sample",
            "extracted_data": {"example_id": "house_vocabulary/sample"},
        },
        "raw_output": {"example_id": "house_vocabulary/sample"},
    }
    (group / "sample.result.json").write_text(json.dumps(result))
    sidecar = {
        "tags": ["challenge:house_vocabulary"],
        "data_schema": {"title": "house_vocabulary"},
        "expected_output": {"group": "house_vocabulary"},
    }
    (group / "sample.test.json").write_text(json.dumps(sidecar))
    (group / "sample.pdf").write_bytes(b"unchanged PDF")
    assert len(migrate_pro_layout(tmp_path)) > 0
    assert group.is_dir()
    migrate_pro_layout(tmp_path, write=True)
    target = tmp_path / "custom_terminology"
    actual = json.loads((target / "sample.result.json").read_text())
    assert actual["request"]["example_id"] == "custom_terminology/sample"
    assert actual["request"]["source_file_path"] == "/data/custom_terminology/sample.pdf"
    assert actual["output"]["example_id"] == "custom_terminology/sample"
    assert actual["output"]["extracted_data"] == result["output"]["extracted_data"]
    assert actual["raw_output"] == result["raw_output"]
    migrated_sidecar = json.loads((target / "sample.test.json").read_text())
    assert migrated_sidecar["tags"] == ["challenge:custom_terminology"]
    assert migrated_sidecar["data_schema"] == sidecar["data_schema"]
    assert migrated_sidecar["expected_output"] == sidecar["expected_output"]
    assert (target / "sample.pdf").read_bytes() == b"unchanged PDF"
    assert migrate_pro_layout(tmp_path, write=True) == ()


def test_conflicting_layout_is_rejected_before_any_file_is_changed(tmp_path: Path):
    """A partially mixed layout must not overwrite results or partly migrate another split."""
    for name in ("house_vocabulary", "custom_terminology", "drawn_relations"):
        (tmp_path / name).mkdir()
    source = tmp_path / "drawn_relations/sample.result.json"
    source.write_text('{"test_id":"drawn_relations/sample"}')
    with pytest.raises(ValueError, match="already exists"):
        migrate_pro_layout(tmp_path, write=True)
    assert source.read_text() == '{"test_id":"drawn_relations/sample"}'
    assert not (tmp_path / "relational_diagrams").exists()


def test_malformed_result_aborts_before_renaming_directories(tmp_path: Path):
    """A corrupt cached result must leave the original layout available for repair."""
    (tmp_path / "drawn_relations").mkdir()
    (tmp_path / "drawn_relations/sample.result.json").write_text('{"truncated":')
    with pytest.raises(ValueError):
        migrate_pro_layout(tmp_path, write=True)
    assert (tmp_path / "drawn_relations").is_dir()
    assert not (tmp_path / "relational_diagrams").exists()


def test_legacy_hf_snapshot_download_reconstructs_canonical_paths_and_tags(tmp_path: Path, monkeypatch):
    """A canonical downloader must read the old JSONL/PDF layout without duplicating or mislabelling cases."""
    import huggingface_hub

    from extract_bench.data.download import dataset_config, download_dataset

    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    for old in LEGACY_PRO_SPLITS:
        document = snapshot / "docs" / old / "sample.pdf"
        document.parent.mkdir(parents=True)
        document.write_bytes(b"source PDF bytes")
        row = {
            "id": f"{old}/sample",
            "category": old,
            "pdf": f"docs/{old}/sample.pdf",
            "data_schema": "{}",
            "expected_output": "{}",
            "tags": [f"challenge:{old}"],
        }
        (snapshot / f"{old}.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(snapshot))
    destination = tmp_path / "download"
    download_dataset(destination, dataset="pro")
    for split in dataset_config("pro").splits:
        assert (destination / split / "sample.pdf").read_bytes() == b"source PDF bytes"
        assert json.loads((destination / split / "sample.test.json").read_text())["tags"] == [f"challenge:{split}"]
    assert {p.name for p in destination.iterdir() if p.is_dir()} == set(dataset_config("pro").splits)


def test_pro_snapshot_stages_public_field_rules(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    for split in LEGACY_PRO_SPLITS:
        (source / f"{split}.jsonl").write_text("", encoding="utf-8")
    for name in ("README.md", "eval.yaml", ".gitattributes"):
        (source / name).write_text("", encoding="utf-8")
    split = "house_vocabulary"
    pdf = source / "docs" / split / "sample.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"source PDF")
    row = {
        "id": f"{split}/sample",
        "category": split,
        "pdf": f"docs/{split}/sample.pdf",
        "tags": [],
        "field_rules": json.dumps({"amount": {"evidence": [{"value": 7, "_repair": "private"}]}}),
    }
    (source / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    stage_pro_snapshot(source, output, "a" * 40)

    staged = json.loads((output / "custom_terminology.jsonl").read_text(encoding="utf-8"))
    assert json.loads(staged["field_rules"]) == {"amount": {"evidence": [{"value": 7}]}}
