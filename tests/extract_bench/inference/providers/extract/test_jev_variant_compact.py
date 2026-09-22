import json

from extract_bench.inference.providers.extract.jev.variant_compact import (
    _REGION_PREFIX,
    _VALUE_SUFFIX,
    CompactClient,
    compact_instructions,
    extract,
)


def test_compact_preserves_description_constraints_and_source():
    schema = {
        "type": "number",
        "title": "Total",
        "description": "Amount in cents. Return null if blank.",
        "minimum": 0,
        "multipleOf": 1,
    }
    source = "Invoice\nTotal: 125\nAdjacent field: 19"
    prompt = f"Extract invoice.total. Field schema: {json.dumps(schema)}\nSOURCE:\n{source}{_VALUE_SUFFIX}"
    compact = compact_instructions(prompt)
    assert source in compact
    assert schema["description"] in compact
    assert '"minimum":0' in compact and '"multipleOf":1' in compact
    assert "invoice.total" in compact
    assert len(compact) < len(prompt)


def test_proxy_preserves_all_candidates_state_and_array_prompts():
    state = {"source": "unmodified"}
    criteria = {f"v{i}": f"value {i}" for i in range(240)}
    array_prompt = "Choose the table containing records for array items. Schema: {}"
    questions = {
        "region": {
            "type": "choice",
            "instructions": _REGION_PREFIX
            + 'name. Field schema: {"type":"string"}. A blank checkbox is still relevant evidence.',
            "criteria": criteria,
        },
        "array": {"type": "choice", "instructions": array_prompt, "criteria": {"none": "Absent"}},
    }

    class Client:
        def decide(self, actual_state, actual_questions):
            assert actual_state is state
            assert actual_questions["region"]["criteria"] is criteria
            assert len(actual_questions["region"]["criteria"]) == 240
            assert actual_questions["array"] == questions["array"]
            return {"region": {"choice": "v0"}}

    assert CompactClient(Client()).decide(state, questions)["region"]["choice"] == "v0"
    assert questions["region"]["instructions"].startswith(_REGION_PREFIX)


def test_localized_end_to_end_and_blank_boolean_rule():
    class Client:
        def decide(self, state, questions):
            answers = {}
            for key, question in questions.items():
                prompt = question["instructions"]
                if prompt.startswith("Where is"):
                    choice = "v0"
                else:
                    assert prompt.startswith("What is")
                    choice = next((k for k, v in question["criteria"].items() if v == '"Ada"'), "none")
                answers[key] = {"choice": choice}
            return answers

    result = extract(
        {"text": "Name: Ada\nApproved [ ]"},
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "approved": {"type": "boolean", "description": "True if checked, false otherwise."},
            },
        },
        Client(),
    )
    assert result["data"] == {"name": "Ada", "approved": False}


def test_unrecognized_or_changed_prompt_is_not_rewritten():
    prompt = 'Extract name. Field schema: {"type":"string"}\nSOURCE:\nSome source with an unfamiliar trailer'
    assert compact_instructions(prompt) == prompt
