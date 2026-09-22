"""Cache LiteParse input without reading extraction labels or calling Jev."""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from extract_bench.inference.providers.extract.jev.parser import parse_document

ROOT = Path(__file__).resolve().parents[2]


def parse(row):
    try:
        document = parse_document(
            ROOT / "data" / row["split"] / (row["id"] + ".pdf"), ROOT / "learning/jev/parse_cache"
        )
        return {
            "id": row["id"],
            "pages": len(document["pages"]),
            "text_chars": len(document["text"]),
            "markdown_chars": len(document.get("markdown", "")),
        }
    except Exception as exc:
        return {"id": row["id"], "error": str(exc)}


if __name__ == "__main__":
    rows = json.loads((ROOT / "learning/luna_round2/inventory.json").read_text())
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(parse, row) for row in rows]):
            results.append(future.result())
            if len(results) % 20 == 0:
                print("PARSED", len(results), "/", len(rows), "errors", sum("error" in r for r in results), flush=True)
    (ROOT / "learning/jev/parse_inventory.json").write_text(json.dumps(results, indent=2))
