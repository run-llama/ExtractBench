"""Summarize saved official evaluation scores, retaining failures as zero."""

import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def main():
    report = {}
    for variant in sorted((HERE / "results").iterdir()):
        if not variant.is_dir():
            continue
        rows = [json.loads(p.read_text()) for p in sorted(variant.glob("*.json"))]
        complete = [r for r in rows if "metrics" in r]
        groups = {}
        for split in ("all", "short", "medium", "long"):
            group = [r for r in complete if split == "all" or r["document"]["split"] == split]
            if group:
                groups[split] = {
                    "n": len(group),
                    "f1": mean(r["metrics"].get("extract_unified_value_f1", 0) for r in group),
                    "failures": sum("error" in r for r in group),
                    "cost": sum(r.get("cost", 0) for r in group),
                }
        report[variant.name] = {
            "groups": groups,
            "documents": [
                {
                    "id": r["document"]["id"],
                    "f1": r["metrics"].get("extract_unified_value_f1", 0),
                    "error": r.get("error"),
                }
                for r in complete
            ],
        }
        print(variant.name, json.dumps(groups))
    (HERE / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
