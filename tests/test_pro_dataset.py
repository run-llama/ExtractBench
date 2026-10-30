"""Pro must remain isolated from original EB through download, loading and run routing."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from extract_bench.cli import BenchCLI
from extract_bench.data.download import MANIFEST_NAME, dataset_config, download_dataset, is_dataset_ready
from extract_bench.test_cases.loader import load_test_cases


@pytest.fixture
def pro_snapshot(tmp_path):
    snapshot = tmp_path / "revision-a"
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    for split in dataset_config("pro").splits:
        pdf = snapshot / "docs" / split / "sample.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(b"%PDF-1.4\n")
        row = {
            "id": f"{split}/sample",
            "category": split,
            "pdf": str(pdf.relative_to(snapshot)),
            "data_schema": json.dumps(schema),
            "expected_output": json.dumps({"answer": 42}),
            "field_rules": "{}",
            "repeated_structure": "{}",
            "tags": [split],
        }
        (snapshot / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    return snapshot


def test_public_cli_download_reconstructs_all_pro_challenges_for_loader(tmp_path, monkeypatch, pro_snapshot):
    monkeypatch.chdir(tmp_path)
    with patch("huggingface_hub.snapshot_download", return_value=str(pro_snapshot)) as snapshot:
        assert BenchCLI().download(dataset="pro") == 0
    assert snapshot.call_args.kwargs["repo_id"] == "llamaindex/ExtractBenchPro"
    assert snapshot.call_args.kwargs["revision"] is None
    assert not Path("data").exists()
    root = Path("data-pro")
    assert BenchCLI().status(dataset="pro") == 0
    cases = load_test_cases(root, product_type="EXTRACT")
    assert {case.group for case in cases} == set(dataset_config("pro").splits)
    assert len(cases) == 6
    assert all(case.expected_output == {"answer": 42} for case in cases)
    assert not is_dataset_ready(root)
    (root / "relational_diagrams/sample.pdf").unlink()
    assert BenchCLI().status(dataset="pro") == 1


@pytest.mark.parametrize("existing,requested", [("extractbench", "pro"), ("pro", "extractbench")])
def test_download_refuses_other_dataset_even_with_force(tmp_path, existing, requested):
    manifest = {"repo": dataset_config(existing).repo, "test": False}
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
    with patch("huggingface_hub.snapshot_download") as snapshot:
        with pytest.raises(RuntimeError, match="belongs to"):
            download_dataset(tmp_path, dataset=requested, force=True)
        snapshot.assert_not_called()
    assert json.loads((tmp_path / MANIFEST_NAME).read_text()) == manifest


@pytest.mark.parametrize("kwargs", [{"dataset": "pro", "test": True}, {"dataset": "typo"}])
def test_invalid_selection_fails_before_download_or_inference(kwargs):
    with patch("huggingface_hub.snapshot_download") as snapshot:
        assert BenchCLI().download(**kwargs) == 1
        snapshot.assert_not_called()
    with patch("extract_bench.pipeline.cli.InferenceCLI") as inference:
        assert BenchCLI().run("dummy", **kwargs) == 1
        inference.assert_not_called()


def test_pro_run_forwards_selection_and_isolates_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli = BenchCLI()
    with (
        patch("extract_bench.pipeline.cli.is_dataset_ready", return_value=False),
        patch("extract_bench.pipeline.cli.download_dataset") as download,
        patch("extract_bench.pipeline.cli.InferenceCLI") as inference,
        patch.object(cli.pipeline, "_run_multi_group_evaluation", return_value=0) as evaluate,
    ):
        inference.return_value.run.return_value = 0
        assert cli.run("dummy", dataset="pro", open_report=False) == 0
    download.assert_called_once_with(data_dir=Path("data-pro"), test=False, dataset="pro")
    assert inference.return_value.run.call_args.kwargs["input_dir"] == Path("data-pro")
    assert inference.return_value.run.call_args.kwargs["output_dir"] == Path("output-pro")
    assert evaluate.call_args.kwargs["input_path"] == Path("data-pro")
    assert evaluate.call_args.kwargs["pipeline_output_dir"] == Path("output-pro/dummy")


def test_pro_refresh_updates_pdf_and_prunes_replaced_documents(tmp_path, pro_snapshot):
    root = tmp_path / "download"
    with patch("huggingface_hub.snapshot_download", return_value=str(pro_snapshot)):
        download_dataset(root, dataset="pro")
    split = "relational_diagrams"
    old = root / split / "removed.test.json"
    old.write_text("{}")
    old.with_name("removed.pdf").write_bytes(b"old")
    new_snapshot = pro_snapshot.rename(pro_snapshot.with_name("revision-b"))
    (new_snapshot / "docs" / split / "sample.pdf").write_bytes(b"%PDF-new")
    with patch("huggingface_hub.snapshot_download", return_value=str(new_snapshot)):
        download_dataset(root, dataset="pro")
    assert (root / split / "sample.pdf").read_bytes() == b"%PDF-new"
    assert not old.exists()
    assert not old.with_name("removed.pdf").exists()
    assert is_dataset_ready(root, "pro")


def test_failed_refresh_cannot_leave_a_ready_manifest(tmp_path, pro_snapshot):
    root = tmp_path / "download"
    with patch("huggingface_hub.snapshot_download", return_value=str(pro_snapshot)):
        download_dataset(root, dataset="pro")
    last_split = pro_snapshot / "long_single_pages.jsonl"
    rows = last_split.read_text()
    last_split.unlink()
    with patch("huggingface_hub.snapshot_download", return_value=str(pro_snapshot)):
        with pytest.raises(RuntimeError, match="source split"):
            download_dataset(root, dataset="pro", force=True)
    assert not is_dataset_ready(root, "pro")

    last_split.write_text(rows)
    (pro_snapshot / "docs/long_single_pages/sample.pdf").write_bytes(b"%PDF-resumed")
    with patch("huggingface_hub.snapshot_download", return_value=str(pro_snapshot)):
        download_dataset(root, dataset="pro")
    assert (root / "long_single_pages/sample.pdf").read_bytes() == b"%PDF-resumed"
    assert is_dataset_ready(root, "pro")


def test_pro_run_refuses_unmigrated_results_before_starting_paid_inference(tmp_path, monkeypatch):
    """Canonical case IDs must not turn existing predictions into cache misses and rerun the provider."""
    monkeypatch.chdir(tmp_path)
    result = tmp_path / "output-pro/dummy/house_vocabulary/sample.result.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"output":{"extracted_data":{"answer":42}}}')
    with (
        patch("extract_bench.pipeline.cli.is_dataset_ready", return_value=True),
        patch("extract_bench.pipeline.cli.InferenceCLI") as inference,
    ):
        inference.return_value.run.return_value = 1
        assert BenchCLI().run("dummy", dataset="pro", open_report=False) == 1
        inference.assert_not_called()
    assert result.exists()


def test_original_pro_group_alias_reaches_both_inference_and_evaluation(tmp_path, monkeypatch):
    """Normalizing only one phase would infer a group whose predictions the evaluator cannot find."""
    monkeypatch.chdir(tmp_path)
    cli = BenchCLI()
    with (
        patch("extract_bench.pipeline.cli.is_dataset_ready", return_value=True),
        patch("extract_bench.pipeline.cli.InferenceCLI") as inference,
        patch.object(cli.pipeline, "_run_evaluation_and_report", return_value=0) as evaluate,
    ):
        inference.return_value.run.return_value = 0
        assert cli.run("dummy", dataset="pro", group="drawn_relations", open_report=False) == 0
    assert inference.return_value.run.call_args.kwargs["group"] == "relational_diagrams"
    assert evaluate.call_args.kwargs["group"] == "relational_diagrams"
