"""Table previews should agree with the document objects scripts read."""

from extract_bench.inference.providers.extract.table_codegen.coverage import coverage_warnings
from extract_bench.inference.providers.extract.table_codegen.document import Page, render_page
from extract_bench.inference.providers.extract.table_codegen.items_document import ItemsDocument, render_items_document


def test_empty_html_table_keeps_later_table_indexes_and_reading_order() -> None:
    page = Page(
        1,
        "<table><tr><td>Apples</td></tr></table>"
        "<p>Before the empty table</p><table></table><p>After the empty table</p>"
        "<table><tr><td>Bananas</td></tr></table>",
    )

    rendered = render_page(page)

    assert [table.table_index for table in page.tables()] == [0, 1, 2]
    assert page.tables()[1].n_rows == 0
    assert (
        rendered.index("Apples")
        < rendered.index("Before the empty table")
        < rendered.index("[page 1 table 1]  (empty)")
        < rendered.index("After the empty table")
        < rendered.index("Bananas")
    )


def test_html_table_without_header_cells_keeps_the_first_row_as_data() -> None:
    page = Page(1, "<table><tr><td>Apples</td><td>8</td></tr><tr><td>Bananas</td><td>5</td></tr></table>")

    rendered = render_page(page)

    assert "header:" not in rendered
    assert "Apples" in rendered
    assert "(2 non-header rows total)" in rendered


def test_stacked_html_header_does_not_relabel_a_typed_data_row() -> None:
    document = ItemsDocument(
        [
            {
                "page_number": 1,
                "items": [
                    {
                        "type": "table",
                        "html": "<table><tr><th>Net</th></tr><tr><th>Amount</th></tr><tr><td>42</td></tr></table>",
                        "rows": [["Net<br/>Amount"], ["42"]],
                    },
                ],
            },
        ],
    )

    rendered = render_items_document(document)

    assert "header: ['Net<br/>Amount']" in rendered
    assert "    ['42']" in rendered
    assert "(1 data rows total)" in rendered


def test_coverage_warning_flags_dropped_and_merged_rows_but_ignores_numeric_column() -> None:
    names = ["Acme Anvils", "Bright Books", "Cedar Cafe", "Delta Diner", "Eagle Electric", "Fox Freight"]
    rows = [["Vendor", "Amount"], *[[name, str(index)] for index, name in enumerate(names, 1)]]
    document = ItemsDocument([{"page_number": 1, "items": [{"type": "table", "rows": rows}]}])
    output = {
        "vendors": [{"name": name, "amount": index} for index, name in enumerate(names[:3], 1)]
        + [{"name": "Delta Diner Eagle Electric"}],
    }

    warning = coverage_warnings(document, output)

    assert all(name in warning for name in names[3:])
    assert all(name not in warning for name in names[:3])
    assert "column 0" in warning
    assert "column 1" not in warning
