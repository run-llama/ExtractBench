"""Command-line interface for data management."""

import sys
from pathlib import Path

from extract_bench.data.download import (
    dataset_config,
    default_data_dir,
    download_dataset,
    is_dataset_ready,
    read_manifest,
)
from extract_bench.data.pro_names import migrate_pro_layout


class DataCLI:
    """Command-line interface for managing the benchmark dataset."""

    def download(
        self,
        data_dir: str | Path | None = None,
        force: bool = False,
        test: bool = False,
        dataset: str = "extractbench",
        revision: str | None = None,
    ) -> int:
        """Download the ExtractBench dataset from HuggingFace.

        Reconstructs the sidecar layout (<split>/<stem>.pdf + <stem>.test.json)
        that inference and evaluation consume.

        Args:
            revision: Optional pinned HF commit, tag, branch, or PR reference.
            data_dir: Local directory to store the dataset
                (default: ./data, or ./data/test when --test is set)
            dataset: "extractbench" (default) or "pro" (./data-pro)
            force: Force re-download even if data already exists
            test: Download the small test dataset (6 documents: 3 short, 2 medium, 1 long)

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            data_path = Path(data_dir) if data_dir else default_data_dir(test=test, dataset=dataset)
            download_dataset(data_dir=data_path, force=force, test=test, dataset=dataset, revision=revision)
            return 0
        except Exception as e:
            print(f"Error downloading dataset: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

    def migrate_pro_names(self, path: str | Path, write: bool = False) -> int:
        """Translate one existing Pro dataset or pipeline directory; dry-run unless --write."""
        try:
            for action in migrate_pro_layout(Path(path), write=write):
                print(action)
            print("Migration applied." if write else "Dry run; pass --write to apply.")
            return 0
        except (ValueError, OSError) as error:
            print(f"Migration refused: {error}", file=sys.stderr)
            return 1

    def status(
        self,
        data_dir: str | Path | None = None,
        test: bool = False,
        dataset: str = "extractbench",
    ) -> int:
        """Check if the dataset is downloaded and show per-split counts.

        Args:
            data_dir: Data directory to check
                (default: ./data, or ./data/test when --test is set)
            test: Check the small test dataset instead of the full dataset
            dataset: "extractbench" (default) or "pro" (./data-pro)

        Returns:
            Exit code (0 if ready, 1 if not)
        """
        data_path = Path(data_dir) if data_dir else default_data_dir(test=test, dataset=dataset)
        if not is_dataset_ready(data_path, dataset=dataset, test=test):
            print(f"Dataset is NOT ready at: {data_path}")
            print(f"Run 'extract-bench download --dataset {dataset}' to download it.")
            return 1

        print(f"Dataset: {data_path}")
        manifest = read_manifest(data_path)
        if manifest is not None and manifest.get("revision") is not None:
            print(f"Revision: {manifest['revision'][:12]}")
        print()
        total = 0
        splits = dataset_config(dataset, test).splits
        width = max(len("Split"), *(len(split) for split in splits))
        hdr = f"{'Split':<{width}} {'Test Cases':>12}"
        print(hdr)
        print("-" * len(hdr))
        for split in sorted(splits):
            n = sum(1 for _ in (data_path / split).rglob("*.test.json"))
            if n == 0:
                continue
            total += n
            print(f"{split:<{width}} {n:>12,}")
        print("-" * len(hdr))
        print(f"{'Total':<{width}} {total:>12,}")
        return 0
