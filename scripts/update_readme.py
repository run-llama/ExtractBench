"""Regenerate README leaderboard tables from leaderboard.csv and leaderboard-pro.csv.

Three tables are generated:

    LEADERBOARD    unified value F1 (the headline metric)
    GROUNDING      word-level and page-level grounding F1, side by side
    PRO_LEADERBOARD  perfect documents, value F1, grounding and cost on Pro

Run: uv run python scripts/update_readme.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "leaderboard.csv"
PRO_CSV_PATH = REPO_ROOT / "leaderboard-pro.csv"
README_PATH = REPO_ROOT / "README.md"


# Grounding is only defined where a field's ground truth carries a verified box,
# so both grounding tables are scored over a smaller and differently distributed
# document set than the value F1 above — their columns do not reconcile with it.
GROUNDING_SCOPE_NOTE = "Scored only over the documents that carry verified box ground truth."


def fmt(v: str) -> str:
    return v if v else "—"


def fmt_cost(v: str) -> str:
    """Cost columns are stored in $/page; render as ¢/page."""
    return f"{float(v) * 100:.2f}¢" if v else "—"


def column_ranks(rows: list[dict], columns: tuple[str, ...]) -> dict[str, dict[float, str]]:
    """Return emphasis for the best and second-best distinct score in each column."""
    ranks = {}
    for column in columns:
        values = sorted({float(row[column]) for row in rows if row[column] != ""}, reverse=True)
        ranks[column] = dict(zip(values[:2], ("best", "second"), strict=False))
    return ranks


def fmt_ranked(value: str, rank: str | None, *, html: bool = False) -> str:
    """Render a score, emphasizing the best and second-best distinct values."""
    rendered = fmt(value)
    if rank == "best":
        return f"<strong>{rendered}</strong>" if html else f"**{rendered}**"
    if rank == "second":
        return f"<u>{rendered}</u>"
    return rendered


def table(header: str, aligns: str, rows: list[list[str]]) -> str:
    return "\n".join([header, aligns, *(f"| {' | '.join(r)} |" for r in rows)])


# README shows only the top N; leaderboard.csv carries the full table.
TOP_N = 10


def value_table(rows: list[dict]) -> str:
    ranked = sorted(
        rows,
        key=lambda r: (
            -float(r["Overall"]),
            float(r["Cost_Per_Page"]) if r["Cost_Per_Page"] else float("inf"),
        ),
    )[:TOP_N]
    score_columns = ("Overall", "Short", "Medium", "Long")
    ranks = column_ranks(rows, score_columns)
    body = table(
        "| Rank | Provider | Category | Overall | Short | Medium | Long | ¢ / Page |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
        [
            [
                str(i),
                r["Provider"],
                r["Category"],
                *(fmt_ranked(r[column], ranks[column].get(float(r[column]))) for column in score_columns),
                fmt_cost(r["Cost_Per_Page"]),
            ]
            for i, r in enumerate(ranked, 1)
        ],
    )
    return f"{body}\n\nTop {TOP_N} of {len(rows)} systems — full table in [leaderboard.csv](leaderboard.csv)."


GROUNDING_PREFIXES = ("Word_Grounding", "Page_Grounding")
GROUNDING_SPANS = ("Word-level grounding F1", "Page-level grounding F1")


def _grounding_cells(row: dict, ranks: dict[str, dict[float, str]]) -> list[str]:
    """The eight grounding numbers, word block then page block."""
    return [
        fmt_ranked(row[key], ranks[key].get(float(row[key])), html=True)
        for prefix in GROUNDING_PREFIXES
        for key in (prefix, f"{prefix}_Short", f"{prefix}_Medium", f"{prefix}_Long")
    ]


def grounding_table(rows: list[dict]) -> str:
    """Show the top grounding providers by overall word-level F1."""
    scoring = [r for r in rows if any(float(r[p]) > 0 for p in GROUNDING_PREFIXES)]
    scoring.sort(key=lambda r: float(r[GROUNDING_PREFIXES[0]]), reverse=True)
    ranked = scoring[:TOP_N]
    score_columns = tuple(
        key
        for prefix in GROUNDING_PREFIXES
        for key in (prefix, f"{prefix}_Short", f"{prefix}_Medium", f"{prefix}_Long")
    )
    ranks = column_ranks(scoring, score_columns)

    span = "".join(f'<th colspan="4">{s}</th>' for s in GROUNDING_SPANS)
    sub = "".join(f'<th align="right">{c}</th>' for c in ("Overall", "Short", "Medium", "Long") * 2)
    body = [
        "    <tr>"
        f'<td align="right">{i}</td><td>{r["Provider"]}</td>'
        + "".join(f'<td align="right">{c}</td>' for c in _grounding_cells(r, ranks))
        + "</tr>"
        for i, r in enumerate(ranked, 1)
    ]
    return "\n".join(
        [
            "<table>",
            "  <thead>",
            f'    <tr><th rowspan="2">Rank</th><th rowspan="2">Provider</th>{span}</tr>',
            f"    <tr>{sub}</tr>",
            "  </thead>",
            "  <tbody>",
            *body,
            "  </tbody>",
            "</table>",
            "",
            f"Top {len(ranked)} of {len(rows)} systems — full table in [leaderboard.csv](leaderboard.csv).",
        ]
    )


def pro_table(rows: list[dict[str, str]]) -> str:
    """Rank the overall Pro slice, retaining challenge rows in the source CSV."""
    overall = [row for row in rows if row["Challenge"] == "all"]
    ranked = sorted(
        overall,
        key=lambda row: (
            -float(row["Perfect_Documents"]),
            -float(row["Value_F1"]),
            float(row["Cost_Per_Page"]) if row["Cost_Per_Page"] != "" else float("inf"),
            row["Provider"],
        ),
    )
    columns = ("Perfect_Documents", "Value_F1", "Word_Grounding", "Page_Grounding")
    ranks = column_ranks(overall, columns)
    return table(
        "| Rank | Provider | Category | Perfect docs (%) | Value F1 | Box F1 | Page F1 | ¢ / Page |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
        [
            [
                str(index),
                row["Provider"],
                row["Category"],
                *(
                    fmt_ranked(row[column], ranks[column].get(float(row[column]))) if row[column] != "" else "—"
                    for column in columns
                ),
                fmt_cost(row["Cost_Per_Page"]),
            ]
            for index, row in enumerate(ranked, 1)
        ],
    )


def replace_block(readme: str, name: str, body: str) -> str:
    start_marker, end_marker = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    start, end = readme.find(start_marker), readme.find(end_marker)
    if start == -1 or end == -1:
        raise SystemExit(f"Markers not found in README.md. Add {start_marker} and {end_marker}.")
    block = f"{start_marker}\n{body}\n{end_marker}"
    return readme[:start] + block + readme[end + len(end_marker) :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if README tables are stale; do not write files")
    args = parser.parse_args()
    with CSV_PATH.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("Provider")]

    with PRO_CSV_PATH.open() as f:
        pro_rows = [row for row in csv.DictReader(f) if row.get("Provider")]

    original = README_PATH.read_text()
    readme = original
    readme = replace_block(
        readme,
        "LEADERBOARD",
        "**Unified value F1** — the headline metric. Every score is an unweighted mean over "
        "documents; each document counts once, whatever its length. For raw data including per-split "
        "precision and recall, cost, and latency, see [leaderboard.csv](leaderboard.csv). "
        "Equal displayed Overall scores are ordered by lower cost per page. "
        "The best score in each Overall, Short, Medium, and Long column is **bold**; the second-best "
        "distinct score is <u>underlined</u>.\n\n" + value_table(rows),
    )
    readme = replace_block(
        readme,
        "GROUNDING",
        "**Grounding F1** — a field counts only when its value is accepted *and* it points at the "
        "right evidence: at word level the predicted box must overlap an accepted evidence box at "
        "IoU 0.5, at page level the cited page must be correct. "
        + GROUNDING_SCOPE_NOTE
        + "\n\n"
        + grounding_table(rows),
    )
    readme = replace_block(
        readme,
        "PRO_LEADERBOARD",
        "Scores are percentages; costs are USD per page, displayed in cents. Rows are ordered by "
        "perfect-document percentage, then Value F1, then lower cost. The best score in each "
        "metric is **bold**; the second-best distinct score is <u>underlined</u>. A dash means "
        "unreported, not zero.\n\n" + pro_table(pro_rows),
    )
    if args.check:
        if readme != original:
            raise SystemExit("README leaderboard tables are stale. Run: uv run python scripts/update_readme.py")
        print("README leaderboard tables are up to date")
    else:
        README_PATH.write_text(readme)
        print(f"Updated README.md from {CSV_PATH.name} and {PRO_CSV_PATH.name}")


if __name__ == "__main__":
    main()
