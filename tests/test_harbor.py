"""Contracts at the JSONL → sandbox → existing evaluator boundaries."""

import json
import runpy
import shutil
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from extract_bench.data.download import dataset_config
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import ValueCell, compute_unified_evidence_metrics
from extract_bench.harbor.export import export_snapshot
from extract_bench.harbor.publish import publish
from extract_bench.harbor.verifier import verify


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "snapshot"
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "object", "properties": {"id": {"type": "string"}, "value": {"type": "integer"}}},
            }
        },
    }
    for split in dataset_config().splits:
        pdf = root / "docs" / split / "sample.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(b"%PDF-1.4\n")
        row = {
            "id": f"{split}/sample",
            "category": split,
            "pdf": str(pdf.relative_to(root)),
            "data_schema": json.dumps(schema),
            "expected_output": json.dumps({"rows": [{"id": "A", "value": 1}, {"id": "B", "value": 2}]}),
            "field_rules": json.dumps({"rows[0].value": {"evidence": [{"value": 7}]}}),
            "repeated_structure": "{}",
            "tags": ["public"],
        }
        (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    (root / "README.md").write_text("---\nconfigs: []\ntags:\n- benchmark\n---\n# Dataset\n")
    (root / ".gitattributes").write_text("*.png filter=lfs diff=lfs merge=lfs -text\n")
    return root


def test_export_keeps_answers_out_of_agent_context_and_preserves_jsonl(tmp_path, snapshot):
    """An over-broad Docker COPY or artifact declaration would expose answers."""
    output = tmp_path / "export"
    first = export_snapshot(snapshot, output, "extractbench", "a" * 40, "b" * 40)
    for split in dataset_config().splits:
        assert (snapshot / f"{split}.jsonl").read_bytes() == (output / f"{split}.jsonl").read_bytes()
        task = next(path for path in (output / "tasks").iterdir() if path.name.startswith(f"{split}--"))
        agent = task / "environment"
        assert {path.name for path in agent.iterdir()} == {"document.pdf", "schema.json", "Dockerfile"}
        assert (agent / "document.pdf").read_bytes() == (snapshot / "docs" / split / "sample.pdf").read_bytes()
        config = tomllib.loads((task / "task.toml").read_text())
        assert config["verifier"]["environment_mode"] == "separate"
        assert config["artifacts"] == ["/app/output.json"]
        assert "COPY document.pdf schema.json /app/" in (agent / "Dockerfile").read_text()
        schema = json.loads((agent / "schema.json").read_text())
        assert schema == json.loads(json.loads((snapshot / f"{split}.jsonl").read_text())["data_schema"])
    # Interrupted reconstruction must repair missing task files on resume.
    short_task = next(path for path in (output / "tasks").iterdir() if path.name.startswith("short--"))
    (short_task / "environment/document.pdf").unlink()
    assert export_snapshot(snapshot, output, "extractbench", "a" * 40, "b" * 40) == first


def test_verifier_honors_acceptable_values_and_order_insensitive_alignment(tmp_path, snapshot):
    """A simplistic equality grader or lost field rules would reject this answer."""
    output = tmp_path / "export"
    export_snapshot(snapshot, output, "extractbench", "a" * 40, "b" * 40)
    short_task = next(path for path in (output / "tasks").iterdir() if path.name.startswith("short--"))
    test_json = short_task / "tests/reference.test.json"
    prediction = tmp_path / "output.json"
    prediction.write_text(json.dumps({"rows": [{"id": "B", "value": 2}, {"id": "A", "value": 7}]}))
    report = verify(test_json, prediction, tmp_path / "logs")
    assert report["value_f1"] == 1.0
    assert report["perfect"] == 1
    feedback = json.loads((tmp_path / "logs/feedback.json").read_text())
    cells = {cell["output_path"]: cell for cell in feedback["cells"]}
    assert cells["rows[1].value"]["reference_path"] == "rows[0].value"
    assert cells["rows[1].value"]["outcome"] == "matched"
    assert sum(cell["f1_credit"] for cell in cells.values()) == pytest.approx(report["value_f1"])
    prediction.write_text(json.dumps({"rows": [{"id": "A", "value": 7}]}))
    assert 0 < verify(test_json, prediction, tmp_path / "logs")["value_f1"] < 1


@pytest.mark.parametrize("submission", [None, "{broken", "[]"])
def test_invalid_submission_scores_zero_without_reusing_previous_reward(tmp_path, submission):
    prediction = tmp_path / "output.json"
    if submission is not None:
        prediction.write_text(submission)
    (tmp_path / "reward.txt").write_text("1")
    report = verify(tmp_path / "reference.test.json", prediction, tmp_path)
    assert report["status"] == "invalid_submission"
    assert float((tmp_path / "reward.txt").read_text()) == 0.0
    feedback = json.loads((tmp_path / "feedback.json").read_text())
    assert feedback["status"] == "invalid_submission"
    assert feedback["cells"] == []


def test_broken_reference_fails_grading_instead_of_assigning_model_zero(tmp_path):
    prediction = tmp_path / "output.json"
    prediction.write_text('{"answer": 42}')
    (tmp_path / "reward.txt").write_text("1")
    (tmp_path / "feedback.json").write_text('{"status": "scored"}')
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(tmp_path / "missing.test.json", prediction, tmp_path)
    assert not (tmp_path / "reward.txt").exists()
    assert not (tmp_path / "feedback.json").exists()


@pytest.mark.parametrize("side", ["reference", "prediction"])
def test_value_cells_cover_nested_unmatched_rows_and_opaque_arrays_without_changing_scores(side):
    """Grounding-only observers lose uncited cells; realigning subtrees can misattribute reordered rows."""
    schema = {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "details": {"type": "object", "properties": {"n": {"type": "integer"}}},
                        "values": {
                            "type": "array",
                            "items": {"type": "object", "properties": {"n": {"type": "integer"}}},
                        },
                    },
                },
            },
            "tags": {"type": "array", "items": {"type": "string"}},
            "defaulted": {"type": "string", "default": "same"},
            "absent": {"type": "string"},
        },
    }
    expected = {
        "groups": [
            {"id": "A", "details": {"n": 1}, "values": [{"n": 10}, {"n": 20}]},
            {"id": "B", "details": {"n": 2}, "values": [{"n": 30}]},
        ],
        "tags": ["x", "y"],
    }
    actual = {
        "groups": [
            {"id": "B", "details": {"n": 2}, "values": [{"n": 30}]},
            {"id": "A", "details": {"n": 999}, "values": [{"n": 20}, {"n": 10}]},
        ],
        "tags": ["wrong"],
    }
    one_sided = {"id": "C", "details": {"n": 3}, "values": [{"n": 40}]}
    (expected if side == "reference" else actual)["groups"].append(one_sided)
    cells: list[ValueCell] = []
    metrics = compute_unified_evidence_metrics(expected, actual, [], data_schema=schema, value_cells=cells)
    assert metrics == compute_unified_evidence_metrics(expected, actual, [], data_schema=schema)
    metadata = next(metric.metadata for metric in metrics if metric.metric_name == "extract_unified_value_f1")
    assert sum(cell.gt_path is not None for cell in cells) == metadata["expected_cells"]
    assert sum(cell.pred_path is not None for cell in cells) == metadata["predicted_cells"]
    assert sum(cell.value_correct for cell in cells) == metadata["v_correct"]
    assert ValueCell("groups[0].values[0].n", "groups[1].values[1].n", True) in cells
    assert ValueCell("groups[0].details.n", "groups[1].details.n", False) in cells
    assert ValueCell("tags", "tags", False) in cells
    assert ValueCell("defaulted", "defaulted", True) in cells
    assert all(cell.gt_path != "absent" and cell.pred_path != "absent" for cell in cells)
    unpaired = [cell for cell in cells if cell.gt_path is None or cell.pred_path is None]
    assert len(unpaired) == 3
    assert all((cell.pred_path is None) == (side == "reference") for cell in unpaired)


def test_feedback_reports_one_sided_cells_and_does_not_assign_credit_to_wrong_values(tmp_path):
    reference = tmp_path / "reference.test.json"
    reference.write_text(
        json.dumps(
            {
                "data_schema": {
                    "type": "object",
                    "properties": {name: {"type": "integer"} for name in ("ok", "wrong", "missing")},
                },
                "expected_output": {"ok": 1, "wrong": 2, "missing": 3},
            }
        )
    )
    prediction = tmp_path / "output.json"
    prediction.write_text(json.dumps({"ok": 1, "wrong": 999, "extra": 4}))
    report = verify(reference, prediction, tmp_path / "logs")
    feedback = json.loads((tmp_path / "logs/feedback.json").read_text())
    cells = {cell["output_path"] or cell["reference_path"]: cell for cell in feedback["cells"]}
    assert {name: cell["outcome"] for name, cell in cells.items()} == {
        "ok": "matched",
        "wrong": "value_differs",
        "missing": "reference_only",
        "extra": "prediction_only",
    }
    assert cells["missing"]["output_path"] is None
    assert cells["extra"]["reference_path"] is None
    assert all(cell["f1_credit"] == 0 for name, cell in cells.items() if name != "ok")
    assert sum(cell["f1_credit"] for cell in cells.values()) == pytest.approx(report["value_f1"])


def test_export_rejects_case_ids_that_escape_tasks_directory(tmp_path, snapshot):
    path = snapshot / "short.jsonl"
    row = json.loads(path.read_text())
    row["id"] = "short/../../escape"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Unsafe"):
        export_snapshot(snapshot, tmp_path / "out", "extractbench", "a" * 40, "b" * 40)


def test_publish_opens_pr_without_overwriting_source_data_and_refuses_modified_tasks(tmp_path, snapshot):
    """An unfiltered upload could replace JSONL or publish unverified task files."""
    output = tmp_path / "export"
    export_snapshot(snapshot, output, "extractbench", "a" * 40, "b" * 40)
    with patch("extract_bench.harbor.publish.HfApi") as api:
        api.return_value.upload_folder.return_value = SimpleNamespace(pr_url="https://hf.example/pr/1")
        assert publish(output) == "https://hf.example/pr/1"
        kwargs = api.return_value.upload_folder.call_args.kwargs
        assert kwargs["create_pr"] is True
        assert kwargs["parent_commit"] == "a" * 40
        assert set(kwargs["allow_patterns"]) == {"README.md", ".gitattributes", "harbor-manifest.json", "tasks/**"}
        api.reset_mock()
        next((output / "tasks").glob("*/instruction.md")).write_text("changed")
        with pytest.raises(ValueError, match="changed after generation"):
            publish(output)
        api.assert_not_called()


@pytest.mark.parametrize(
    "damaged", ["environment/schema.json", "environment/document.pdf", "tests/reference.test.json"]
)
def test_smoke_rejects_exported_inputs_or_references_that_disagree_with_the_hf_row(tmp_path, snapshot, damaged):
    smoke = runpy.run_path(str(Path(__file__).parents[1] / "scripts/smoke_harbor.py"))
    output = tmp_path / "export"
    manifest = export_snapshot(snapshot, output, "extractbench", "a" * 40, "b" * 40)
    task_id = next(iter(manifest["tasks"]))
    task = output / "tasks" / task_id
    row = json.loads((snapshot / "short.jsonl").read_text())
    smoke["validate_task"](task, row, snapshot / row["pdf"])
    (task / damaged).write_bytes(b"{}" if damaged.endswith(".json") else b"%PDF-another-document")
    with pytest.raises(ValueError, match="differs"):
        smoke["validate_task"](task, row, snapshot / row["pdf"])


def test_smoke_rejects_published_tasks_when_source_jsonl_changed_without_regeneration(tmp_path, snapshot):
    smoke = runpy.run_path(str(Path(__file__).parents[1] / "scripts/smoke_harbor.py"))
    output = tmp_path / "published"
    export_snapshot(snapshot, output, "extractbench", "a" * 40, "b" * 40)
    shutil.copytree(snapshot / "docs", output / "docs")
    files = [f"{split}.jsonl" for split in dataset_config().splits]
    rows = {row["id"]: row for file in files for row in [json.loads((snapshot / file).read_text())]}
    smoke["validate_published"](output, rows, files)
    path = output / "short.jsonl"
    row = json.loads(path.read_text())
    row["expected_output"] = json.dumps({"rows": []})
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="source hashes"):
        smoke["validate_published"](output, rows, files)


def test_smoke_container_build_context_uses_the_wheel_built_from_the_checkout(tmp_path):
    """A stale image/package install would exercise a different scorer than the checkout."""
    smoke = runpy.run_path(str(Path(__file__).parents[1] / "scripts/smoke_harbor.py"))
    source = tmp_path / "task"
    (source / "tests").mkdir(parents=True)
    wheel = tmp_path / "checkout_build-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"checkout wheel contents")
    target = tmp_path / "container-task"
    smoke["container_task"](source, target, wheel)
    assert (target / "tests" / wheel.name).read_bytes() == wheel.read_bytes()
    assert f'"/tests/{wheel.name}"' in (target / "tests/Dockerfile").read_text()
