"""Contracts at the JSONL → sandbox → existing evaluator boundaries."""

import json
import tomllib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from extract_bench.data.download import dataset_config
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


def test_broken_reference_fails_grading_instead_of_assigning_model_zero(tmp_path):
    prediction = tmp_path / "output.json"
    prediction.write_text('{"answer": 42}')
    (tmp_path / "reward.txt").write_text("1")
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(tmp_path / "missing.test.json", prediction, tmp_path)
    assert not (tmp_path / "reward.txt").exists()


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
