from extract_bench.inference.providers.extract.jev import variant_rowrepair as variant


class Client:
    def __init__(self):
        self.repairs = 0
        self.repair_batches = 0

    def decide(self, state, questions):
        answers = {}
        self.repair_batches += any(key.startswith("repair") for key in questions)
        for key, question in questions.items():
            instructions = question["instructions"]
            if key.startswith("r") and not key.startswith("repair"):
                assert "none" not in question["criteria"]
                choice = "c2" if "Name" in instructions else "c0"
            elif "Extract only from this row:" in instructions:
                self.repairs += 1
                if ".count;" in instructions:
                    choice = "none"
                else:
                    choice = "c0"
            else:
                choice = "c" + key[1:]
            answers[key] = {"choice": choice}
        return answers


def test_right_aligned_numbers_and_all_pages(monkeypatch):
    monkeypatch.setattr(variant.variant_localized, "extract", lambda *args: {"data": {}, "evidence": []})
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "object", "properties": {"name": {"type": "string"}, "amount": {"type": "number"}}},
            }
        },
    }
    pages = [
        {
            "page_index": p,
            "text": "\n".join(
                ["Name       Amount"] + [f"Item{i:<4}    {i:10}.00" for i in range(p * 70 + 1, p * 70 + 71)]
            ),
        }
        for p in range(2)
    ]
    client = Client()
    result = variant.extract({"pages": pages}, schema, client)
    assert len(result["data"]["rows"]) == 140
    assert result["data"]["rows"][-1]["amount"] == 140
    assert client.repairs == 0


def test_ragged_rows_repair_instead_of_shifting_missing_column(monkeypatch):
    monkeypatch.setattr(variant.variant_localized, "extract", lambda *args: {"data": {}, "evidence": []})
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "count": {"type": "integer"},
                        "amount": {"type": "number"},
                    },
                },
            }
        },
    }
    document = {"text": "Name    Count    Amount\nA       1        10\nB       2        20\nMissing          9"}
    client = Client()
    result = variant.extract(document, schema, client)
    assert result["data"]["rows"][-1] == {"name": "Missing", "count": None, "amount": 9}
    assert client.repairs == 3


def test_repairs_batched_across_rows(monkeypatch):
    monkeypatch.setattr(variant.variant_localized, "extract", lambda *args: {"data": {}, "evidence": []})
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "count": {"type": "integer"},
                        "amount": {"type": "number"},
                    },
                },
            }
        },
    }
    text = "\n".join(["Name    Count    Amount"] + ["A       1        10"] * 60 + ["Missing          9"] * 40)
    client = Client()
    result = variant.extract({"text": text}, schema, client)
    assert len(result["data"]["rows"]) == 100
    assert client.repairs == 120
    assert client.repair_batches == 1


def test_complete_blank_string_and_numeric_cells_stay_null_without_repairs(monkeypatch):
    monkeypatch.setattr(variant.variant_localized, "extract", lambda *args: {"data": {}, "evidence": []})
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "optional_text": {"type": "string"},
                        "optional_number": {"type": "number"},
                    },
                },
            }
        },
    }
    client = Client()
    result = variant.extract(
        {"text": "| Name | Optional text | Optional number |\n| Alpha | | |\n| Beta | | |"}, schema, client
    )
    assert result["data"]["rows"] == [
        {"name": "Alpha", "optional_text": None, "optional_number": None},
        {"name": "Beta", "optional_text": None, "optional_number": None},
    ]
    assert client.repairs == 0
    assert client.repair_batches == 0
