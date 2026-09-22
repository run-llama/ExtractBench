from extract_bench.inference.providers.extract.jev import variant_anchors as variant
from extract_bench.inference.providers.extract.jev.variant_rowrepair import source_rows


def test_anchor_groups_wrapped_text_without_losing_records():
    rows = source_rows(
        {
            "text": "Issuer    Type           ID\nAlpha     Common         123456789\n          Stock\nBeta      Preferred      987654321\n          Stock"  # noqa: E501
        }
    )[0]
    anchor = next(c for c in variant.anchor_candidates(rows) if c["family"] == "nine-digit identifier")
    lines, merged = variant.group_rows(rows, anchor)
    assert merged == 2
    assert len(lines) == 3
    assert "Common Stock" in lines[1]
    assert "Preferred Stock" in lines[2]


def test_markdown_tables_preferred_with_no_page_record_cap(monkeypatch):
    monkeypatch.setattr(variant.variant_localized, "extract", lambda *args: {"data": {}, "evidence": []})
    captured = []

    def downstream(document, schema, client):
        captured.append(document)
        return {"data": {"holdings": []}, "evidence": []}

    monkeypatch.setattr(variant.variant_rowrepair, "extract", downstream)

    class NoCalls:
        def decide(self, *args):
            raise AssertionError("Markdown tables do not need anchor selection")

    pages = [
        {
            "page_index": p,
            "text": "Broken source",
            "markdown": "\n".join(
                ["| Name | Empty | Amount |", "| --- | --- | --- |"]
                + [f"| Item{p * 70 + i} | | {i} |" for i in range(70)]
            ),
        }
        for p in range(2)
    ]
    schema = {
        "type": "object",
        "properties": {
            "holdings": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}}}}
        },
    }
    variant.extract({"pages": pages}, schema, NoCalls())
    assert len(captured[0]["pages"]) == 2
    assert "Item139 | | 69" in captured[0]["text"]
    assert "Broken source" not in captured[0]["text"]


def test_identical_explicit_headers_reuse_mapping():
    class Client:
        calls = 0

        def decide(self, state, questions):
            self.calls += 1
            return {"f0": {"choice": "c0"}}

    client = Client()
    cached = variant.MappingCache(client, [{"text": "| Name | Amount |"}])
    questions = {"f0": {"instructions": "Which name?", "criteria": {"c0": "column sample", "none": "absent"}}}
    cached.decide("Header and source:\n| Name | Amount |\nA | 1\nSample complete rows:A", questions)
    cached.decide("Header and source:\n| Name | Amount |\nB | 2\nSample complete rows:B", questions)
    assert client.calls == 1
