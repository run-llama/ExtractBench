from extract_bench.inference.providers.extract.jev.variant_multipage import extract, regions


def test_all_rows_across_multiple_pages():
    class Client:
        def decide(self, state, questions):
            answers = {}
            for key, question in questions.items():
                if key == "relevant":
                    choice = "c0"
                elif key.startswith("field"):
                    choice = "c" + key.removeprefix("field")
                else:
                    choice = "c2" if "Name | Amount" in question["instructions"] else "c0"
                answers[key] = {"choice": choice}
            return answers

    pages = [
        {"page_index": page, "text": "\n".join(["Name | Amount"] + [f"Item{page * 70 + r} | {r}" for r in range(70)])}
        for page in range(2)
    ]
    schema = {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "items": {"type": "object", "properties": {"name": {"type": "string"}, "amount": {"type": "number"}}},
            }
        },
    }
    result = extract({"pages": pages}, schema, Client())
    assert len(result["data"]["records"]) == 140
    assert result["data"]["records"][-1] == {"name": "Item139", "amount": 69}
    assert result["evidence"][-1]["page_index"] == 1


def test_geometry_retains_empty_middle_column():
    words = []
    for y, cells in [
        (0, [(0, "Name"), (100, "Count"), (200, "Price")]),
        (12, [(0, "A"), (100, "2"), (200, "10")]),
        (24, [(0, "B"), (200, "20")]),
    ]:
        words.extend({"x": x, "y": y, "width": 20, "height": 10, "text": text} for x, text in cells)
    found = regions({"pages": [{"page_index": 0, "width": 300, "words": words}]})
    assert found[0].rows[-1] == ["B", "", "20"]
