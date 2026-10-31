"""Pro must remain isolated from original EB through download, loading and run routing."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

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
    original = tmp_path / "house_vocabulary/sample.test.json"
    original.parent.mkdir()
    original.write_text('{"test_id":"house_vocabulary/sample"}', encoding="utf-8")
    with patch("huggingface_hub.snapshot_download") as snapshot:
        with pytest.raises(RuntimeError, match="belongs to"):
            download_dataset(tmp_path, dataset=requested, force=True)
        snapshot.assert_not_called()
    assert json.loads((tmp_path / MANIFEST_NAME).read_text()) == manifest
    assert original.read_text(encoding="utf-8") == '{"test_id":"house_vocabulary/sample"}'


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


def test_pro_run_migrates_dataset_and_reuses_prediction_without_calling_provider(tmp_path, monkeypatch) -> None:
    """Renamed inputs and outputs must meet at the real runner's cache before any model call."""
    from extract_bench.inference.pipelines import get_pipeline
    from extract_bench.inference.providers.base import Provider
    from extract_bench.schemas.extract_output import ExtractOutput
    from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "data-pro/house_vocabulary/sample.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.4\n")
    sidecar = {
        "data_schema": {"type": "object", "properties": {"answer": {"type": "integer"}}},
        "expected_output": {"answer": 42},
        "tags": ["challenge:house_vocabulary"],
    }
    source.with_suffix(".test.json").write_text(json.dumps(sidecar), encoding="utf-8")
    pipeline = get_pipeline("llamaextract_agentic")
    now = datetime.now(UTC)
    payload = InferenceResult(
        request=InferenceRequest(
            example_id="house_vocabulary/sample", source_file_path=str(source), product_type="extract"
        ),
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type="extract",
        output=ExtractOutput(
            example_id="house_vocabulary/sample", pipeline_name=pipeline.pipeline_name, extracted_data={"answer": 42}
        ),
        raw_output={"example_id": "house_vocabulary/sample"},
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    result = tmp_path / "output-pro" / pipeline.pipeline_name / "house_vocabulary/sample.result.json"
    result.parent.mkdir(parents=True)
    result.write_text(payload.model_dump_json(), encoding="utf-8")
    provider = Mock(spec=Provider)
    with patch("extract_bench.inference.cli.create_provider", return_value=provider):
        assert (
            BenchCLI().run(
                pipeline.pipeline_name,
                input_dir="data-pro",
                dataset="pro",
                group="house_vocabulary",
                max_concurrent=1,
                verbose=True,
                open_report=False,
            )
            == 0
        )
    provider.run_inference.assert_not_called()
    provider.normalize.assert_not_called()
    current = result.parent.parent / "custom_terminology/sample.result.json"
    actual = json.loads(current.read_text(encoding="utf-8"))
    assert actual["request"]["example_id"] == "custom_terminology/sample"
    assert actual["request"]["source_file_path"] == str(tmp_path / "data-pro/custom_terminology/sample.pdf")
    assert actual["output"]["extracted_data"] == {"answer": 42}
    assert actual["raw_output"] == payload.raw_output
    assert (current.parent.parent / "_evaluation_report_detailed.html").is_file()


def test_download_migrates_original_layout_without_downloading_it_again(tmp_path, pro_snapshot) -> None:
    """A complete original download must become ready without fetching or reconstructing its PDFs."""
    from extract_bench.data.pro_names import LEGACY_PRO_SPLITS

    root = tmp_path / "data-pro"
    with patch("huggingface_hub.snapshot_download", return_value=str(pro_snapshot)):
        download_dataset(root, dataset="pro")
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    for old, new in LEGACY_PRO_SPLITS.items():
        (root / new).rename(root / old)
        manifest["per_split"][old] = manifest["per_split"].pop(new)
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with (
        patch("huggingface_hub.HfApi") as api,
        patch(
            "huggingface_hub.snapshot_download", side_effect=AssertionError("Complete legacy download fetched again")
        ) as snapshot,
    ):
        api.return_value.repo_info.return_value.sha = manifest["revision"]
        assert download_dataset(root, dataset="pro") == root
        snapshot.assert_not_called()
    assert is_dataset_ready(root, "pro")
    assert all((root / split / "sample.pdf").read_bytes() == b"%PDF-1.4\n" for split in dataset_config("pro").splits)


@pytest.mark.parametrize("damage", ["collision", "malformed"])
def test_unsafe_result_migration_stops_before_provider_creation(tmp_path, pro_snapshot, damage) -> None:
    """Conflicting or malformed cached results must survive unchanged without starting inference."""
    result = tmp_path / "output/llamaextract_agentic/house_vocabulary/sample.result.json"
    result.parent.mkdir(parents=True)
    original = '{"truncated":' if damage == "malformed" else '{"request":{},"output":{}}'
    result.write_text(original, encoding="utf-8")
    if damage == "collision":
        (result.parent.parent / "custom_terminology").mkdir()
    with patch("extract_bench.inference.cli.create_provider") as provider:
        assert (
            BenchCLI().run(
                "llamaextract_agentic",
                input_dir=pro_snapshot,
                output_dir=tmp_path / "output",
                dataset="pro",
                max_concurrent=1,
                open_report=False,
            )
            == 1
        )
        provider.assert_not_called()
    assert result.read_text(encoding="utf-8") == original


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
