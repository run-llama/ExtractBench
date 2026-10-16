# ExtractBench Pro leaderboard

The [README table](../README.md#extractbench-pro-leaderboard) summarizes [leaderboard-pro.csv](../leaderboard-pro.csv). The CSV contains overall results and six challenge slices for each system. To submit or update scores, see [Contributing](../CONTRIBUTING.md#submitting-benchmark-results).

## Scoring

- **Perfect documents:** percentage of documents with 100% Value F1, before rounding the displayed score. Use a tolerance of `1e-9` on the scorer's 0–1 F1 scale.
- **Value F1:** unweighted mean of `extract_unified_value_f1` over documents.
- **Box F1:** unweighted mean of `extract_unified_grounded_f1`, requiring both the correct value and a correct evidence box at IoU 0.5. This is not Grounded AUC F1-IoU.
- **Page F1:** unweighted mean of `extract_unified_page_f1`, requiring the correct value and source page.

The leaderboard applies a 30-minute limit per document. Failed, missing, and over-limit documents contribute zero rather than disappearing from the denominator. Grounding scores cover only documents with verified evidence boxes; the official cohort has 100 value-scored documents and 81 grounding-scored documents. Overall scores are document-weighted, not the average of the six challenge means.

| Challenge key | Documents | Grounding documents |
|---|---:|---:|
| `document_conventions` | 22 | 22 |
| `house_vocabulary` | 14 | 14 |
| `house_counting_rules` | 12 | 9 |
| `drawn_relations` | 20 | 19 |
| `charts_inside_documents` | 25 | 10 |
| `long_chart_sheets` | 7 | 7 |
| `all` | 100 | 81 |

## CSV format

Each `(Provider, Challenge)` pair has one row. `all` is the overall result; other challenge keys match the download directories.

| Column | Meaning |
|---|---|
| `Provider` | System name, including model/configuration where relevant |
| `Category` | LlamaExtract, Specialized APIs, or Coding Agents |
| `Challenge` | `all` or one of the six challenge keys above |
| `Documents` | Full value-scoring denominator, including failures |
| `Grounding_Documents` | Denominator for box and page grounding |
| `Perfect_Documents` | Percentage of documents with perfect Value F1 |
| `Value_F1` | Value F1 on a 0–100 scale |
| `Word_Grounding` | Box F1 on a 0–100 scale; name matches original ExtractBench's CSV |
| `Page_Grounding` | Page F1 on a 0–100 scale |
| `Cost_Per_Page` | USD per page for this slice; the README converts to cents |

Blank grounding metrics and costs mean unreported, not zero. Perfect-document and Value F1 scores are required for the overall ranking. Rows are ordered by perfect-document percentage, then Value F1, then lower cost. Rank emphasis uses each metric independently.

## Launch snapshot

The initial twelve systems reproduce the 30-minute-limit results published in the [ExtractBench Pro launch post](https://www.llamaindex.ai/blog/introducing-extractbench-pro) and [benchmark website](https://extractbench.ai/pro/). The launch snapshot was assembled on October 14, 2026 and covers the released 100-document, 2,622-page cohort. Scores and costs retain the published precision; costs are the launch estimates, rather than newly measured API bills. No new inference is required to reproduce these table entries.

The CSV is the editable source for this repository's table. `scripts/update_readme.py` renders it alongside the original ExtractBench tables; CI checks that generated README sections match both CSVs.
