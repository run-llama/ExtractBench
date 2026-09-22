# Reproducing LiteParse + Jev

Run the commands below from the repository root. Use the standard registered
provider for fresh inference. The research freeze records the historical
experiment; it is not a portable execution manifest for this newer upstream
checkout.

## Fresh inference

```bash
uv sync --extra jev
uv run extract-bench download
uv run extract-bench pipelines
```

Set `OPENROUTER_API_KEY` in `.env`, then run:

```bash
uv run extract-bench run jev_liteparse_router --max_concurrent=4 --open_report=False
```

An optional small smoke run uses `--test`. A full run covers 370 documents across
`short`, `medium`, and `long`. Results use the normal
`output/jev_liteparse_router/<split>/<id>.result.json` layout. Inference uses
LiteParse 2.14.4 and the OpenRouter Decisions API model alias
`typesafe/jev-1.13`; the historical run resolved that alias to
`typesafe/jev-1.13-20260917`. A future alias resolution or API response may differ.

The router chooses anchors when the parsed Markdown has at least 100 nonempty,
nonseparator pipe rows and the schema contains an object array; otherwise it
chooses compact. Each document runs exactly one selected engine. Routing does
not inspect reference answers or document identifiers.

The provider keeps parser caches, request/response archives, and a persistent
spend ledger in `output/jev_liteparse/`. Its default $9 cap is cumulative for all
pipelines using that artifact directory, including uncertain failed-request
reservations. Re-running the CLI resumes existing results by default. Preserve
the ledger when resuming; a different result `--output_dir` alone does not create
a new spending budget. Provider configuration exposes `artifact_directory` and
`budget_usd` when an independently budgeted run is required.

## Re-evaluate archived predictions without model calls

The submitted archive is `learning/jev/final/native_results.jsonl.gz`. Each line
is a native `InferenceResult` JSON object, including the extracted data,
citations, public schema, and operational statistics. It is a transport format:
the official evaluation runner discovers individual `*.result.json` files,
not JSONL. Download the dataset first, then unpack:

```bash
uv run python - <<'PY'
import gzip
import json
from pathlib import Path

pipeline = "jev_liteparse_router"
root = Path("output_archived") / pipeline
assert not root.exists(), "Use a fresh destination to avoid mixing runs"
failures = []
root.mkdir(parents=True, exist_ok=True)
with gzip.open("learning/jev/final/native_results.jsonl.gz", "rt", encoding="utf-8") as archive:
    for line in archive:
        result = json.loads(line)
        source = Path(result["request"]["source_file_path"])
        split = source.parent.name
        assert split in {"short", "medium", "long"}, source
        assert result["pipeline_name"] == pipeline
        local_pdf = Path("data") / split / source.name
        assert local_pdf.is_file(), local_pdf
        result["request"]["source_file_path"] = str(local_pdf.resolve())
        # Grouped dataset discovery uses split/id, not the bare archive id.
        example_id = f"{split}/{local_pdf.stem}"
        result["request"]["example_id"] = example_id
        result["output"]["example_id"] = example_id
        if result.get("raw_output", {}).get("inference_failed"):
            failures.append({"example_id": result["request"]["example_id"],
                             "error": "Archived inference failure"})
            continue
        target = root / split / (local_pdf.stem + ".result.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
(root / "_errors.json").write_text(json.dumps(failures), encoding="utf-8")
(root / "_metadata.json").write_text(json.dumps({
    "pipeline_name": pipeline,
    "test_cases_dir": str(Path("data").resolve()),
    "product_type": "extract",
}), encoding="utf-8")
PY

uv run extract-bench evaluation run output_archived/jev_liteparse_router \
  --test_cases_dir=data --product_type=extract \
  --pipeline_name=jev_liteparse_router --force=True --max_workers=4
```

The unpack step remaps source PDF paths and qualifies request/output example IDs
as `<split>/<id>` to match grouped dataset discovery; it preserves predictions
and citations. The archive unpacks into a fresh
`output_archived/` directory, separate from fresh inference results. Keep the final
pipeline folder named `jev_liteparse_router`, because the evaluator filters exact path
components. The archive contains all 370 document entries, including recorded
inference failures. Failed transport rows have empty-output placeholders and
`raw_output.inference_failed=True`. The unpacker writes these to `_errors.json`
instead of creating prediction files, so the official runner assigns zero rather
than awarding null/default-field matches. Do not score those placeholders as
successful empty extractions.

For per-split reports, repeat evaluation with `--group=short` and
`--report_dir=output_archived/jev_liteparse_router/short`; substitute `medium` and `long`
for the other splits. The evaluator writes `_evaluation_report.json` and CSV,
Markdown, and HTML reports. These commands make no model calls.

Historical scores use upstream evaluator commit
`11937d102cad9d9b815044888d55e7ce343d3924` and `parse-bench==1.0.4`.
Retain this evaluator revision and dependency version to reproduce those scores;
newer evaluators can legitimately produce different results. Dataset and input
artifact hashes are recorded in `learning/jev/final/report.json`.

### Official failure aggregation

The pinned batch runner adds every observed grounding metric to every failed
document with value zero, including failures without eligible grounding annotations.
The submitted report follows this unchanged behavior: word/page denominators are
239/295 for the budget-limited run. All 20 value/grounding means across the overall
and three split reports match the official native-artifact roundtrip exactly.
Cost figures come from the per-document ledger reconciliation, including charges
incurred on failed documents; quality rescoring does not reconstruct those charges.

## Source-freeze compatibility

The research run started from local ExtractBench commit
`5d04ac6d62a98c6864ac37c21a832eacdfd6240e`. Its freeze includes
`src/extract_bench/inference/providers/extract/table_codegen/schema_utils.py`:

| Source | SHA-256 |
| --- | --- |
| Historical helper, archived in `learning/jev/frozen_sources/schema_utils.py` | `0d142666f01b497c2b17beba54200f2b3f9efeba92c7ee51ec90b6bd1543f075` |
| Helper in upstream evaluator/submission base | `0c18e0ac869428b307ae368099825f094f6ff3624722ca879e2013c49f81d718` |

The helpers differ in documentation and one added upstream function:
`effective_schema(schema)`, a wrapper returning `_effective(schema)`. Every
pre-existing function has an identical Python AST, including the `_effective`
and `resolve_refs` functions imported by Jev variants. This preserves their
extraction semantics while retaining the upstream module unchanged.

The original frozen research runner will reject the newer helper's byte hash.
Do not overwrite the upstream helper or rewrite the historical freeze to bypass
that check. Fresh runs use the official provider commands above; the historical
freeze and archived helper support auditing the original experiment. Research
reporting scripts may additionally reference the original development inventory;
they are not required for native prediction rescoring.

Verify the archived helper with:

```bash
sha256sum learning/jev/frozen_sources/schema_utils.py
```

## Leaderboard artifacts

`learning/jev/final/proposed_leaderboard_row.csv` contains the proposed row;
`per_document.csv` and `report.json` expose document-level results and provenance.
The official CSV stores value/precision/recall/grounding scores as percentages,
costs as USD per page, and latency as seconds per document. Overall value F1 is
the unweighted document mean, including failures; grounding uses its eligible
document subsets. Cost per page is the mean of each document's cost divided by
its page count, not total cost divided by total pages. Preserve unavailable
latency as unavailable rather than filling it with zero.

After reviewing and adding the proposed row to the root `leaderboard.csv`,
regenerate the existing README table blocks:

```bash
uv run python scripts/update_readme.py
```

The README displays only the top ten systems but retains the full system count.
`uv run extract-bench leaderboard` generates a local comparison HTML page; it
does not update or publish the root CSV. Maintainer acceptance is separate from
preparing these artifacts.
