"""Stage the canonical Pro release locally; remote publication is a separate review step."""

import argparse
from pathlib import Path

from extract_bench.data.pro_release import stage_pro_snapshot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    result = stage_pro_snapshot(args.source, args.output, args.source_revision)
    print(f"Staged {len(result['cases'])} cases; no remote changes made")
