from extract_bench.inference.providers.extract.jev.citations import build_citations


def page(lines, index=0):
    words = []
    for row, line in enumerate(lines):
        for col, text in enumerate(line.split()):
            words.append({"text": text, "x": col * 30, "y": row * 20, "width": 20, "height": 10})
    return {"page_index": index, "text": "\n".join(lines), "words": words, "width": 200, "height": 100}


def cite(value, evidence, pages):
    return build_citations(
        {"data": {"rows": [{"value": value}]}, "evidence": [{"path": "rows.0.value.", **evidence}]}, {"pages": pages}
    )


def test_local_context_resolves_repeated_value_and_normalizes_path():
    result = cite(42, {"text": "Total 42", "page_index": 0}, [page(["Tax 42", "Total 42"])])
    assert len(result) == 1
    assert result[0]["field_path"] == "rows[0].value"
    assert result[0]["bbox"] == [0.15, 0.2, 0.1, 0.1]


def test_repeated_value_without_local_context_is_page_only():
    result = cite(42, {"source_text": "42"}, [page(["Tax 42", "Total 42"])])
    assert len(result) == 1 and "bbox" not in result[0]


def test_physical_row_resolves_repeated_value():
    result = cite(42, {"source_text": "42", "page_index": 0, "y": 20}, [page(["Tax 42", "Total 42"])])
    assert result[0]["bbox"][1] == 0.2


def test_ambiguous_page_is_not_guessed():
    assert cite(42, {"source_text": "42"}, [page(["Tax 42"]), page(["Total 42"], 1)]) == []


def test_declared_page_does_not_override_missing_evidence():
    assert cite(42, {"source_text": "Total 42", "page_index": 0}, [page(["Total 17"]), page(["Total 42"], 1)]) == []


def test_localized_context_metadata_selects_actual_page():
    result = cite(
        42,
        {"source_text": 42, "context": "Page 2, lines around 0:\nTotal 42"},
        [page(["Total 42"]), page(["Total 42"], 1)],
    )
    assert result[0]["page"] == 2 and "bbox" in result[0]


def test_accounting_number_and_zero_have_grounding():
    assert cite(-1234.5, {"source_text": "($1,234.50)"}, [page(["($1,234.50)"])])[0]["bbox"]
    assert cite(0, {"source_text": 0}, [page(["0"])])[0]["bbox"]


def test_sign_and_digit_substrings_do_not_match():
    assert cite(12, {"source_text": "-12"}, [page(["-12"])]) == []
    assert cite(12, {"source_text": "120"}, [page(["120"])]) == []
    assert cite("12", {"source_text": "120"}, [page(["120"])]) == []


def test_invalid_dimensions_keep_page_only_and_duplicate_evidence_dedupes():
    source = page(["Total 42"])
    source["width"] = 0
    result = cite(42, {"source_text": "42"}, [source])
    assert len(result) == 1 and "bbox" not in result[0]
    prediction = {"data": {"value": 42}, "evidence": [{"path": "value", "text": "Total 42"}] * 2}
    assert len(build_citations(prediction, {"pages": [page(["Total 42"])]})) == 1


def test_page_and_span_caches_preserve_row_specific_disambiguation(monkeypatch):
    from extract_bench.inference.providers.extract.jev import citations as module

    original_words, original_spans = module._ordered_words, module._spans
    calls = {"words": 0, "spans": 0}

    def words(*args):
        calls["words"] += 1
        return original_words(*args)

    def spans(*args):
        calls["spans"] += 1
        return original_spans(*args)

    monkeypatch.setattr(module, "_ordered_words", words)
    monkeypatch.setattr(module, "_spans", spans)
    prediction = {
        "data": {"rows": [{"value": 0} for _ in range(20)]},
        "evidence": [
            {"path": f"rows[{i}].value", "source_text": "0", "page_index": 0, "y": 20 * (i % 2)} for i in range(20)
        ],
    }
    result = module.build_citations(prediction, {"pages": [page(["0", "0"])]})
    assert len(result) == 20
    assert [item["bbox"][1] for item in result] == [0, 0.2] * 10
    assert calls == {"words": 1, "spans": 3}
