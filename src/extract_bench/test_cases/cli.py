"""Fire CLI group for ``extract-bench dataset``.

snake_case methods become subcommands, every method returns ``int``, guard
failures print to ``sys.stderr`` and return 1, and ``extract_bench`` imports
live inside method bodies so the top-level CLI stays import-light.
"""

from __future__ import annotations

import sys

__all__ = ["DatasetCLI"]

_FORMATS = ("text", "json")


class DatasetCLI:
    """Commands that gate a test-case tree before it ships."""

    def lint(self, test_cases_dir: str, format: str = "text", allow_legacy_rules: bool = False) -> int:
        """Lint every extract ``*.test.json`` under a directory; exit 1 on any finding.

        Sidecars without ``data_schema`` (parse / layout tests share the suffix)
        are skipped, not failed. Exit 2 when the tree holds no extract sidecar
        at all, so an empty or mistyped path cannot pass as clean.

        Args:
            test_cases_dir: Dataset root; recursed.
            format: ``text`` (one line per finding plus a summary) or ``json``.
            allow_legacy_rules: Tolerate the legacy ``test_rules`` list container.
                For auditing an unmigrated tree only; a writer never gets this.
        """
        import json
        from pathlib import Path

        from extract_bench.test_cases.extract_gt_lint import LintFinding, lint_extract_sidecar

        if format not in _FORMATS:
            print(f"Error: unknown format {format!r}; expected one of {', '.join(_FORMATS)}.", file=sys.stderr)
            return 1
        root = Path(test_cases_dir)
        if not root.is_dir():
            print(f"Error: {root} is not a directory.", file=sys.stderr)
            return 1

        rows: list[dict[str, str]] = []
        linted = 0
        skipped = 0
        failed_files = 0
        for path in sorted(root.rglob("*.test.json")):
            rel = str(path.relative_to(root))
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                findings = [LintFinding("file.unreadable", rel, str(exc))]
                payload = None
            if payload is not None:
                if not isinstance(payload, dict) or "data_schema" not in payload:
                    skipped += 1
                    continue
                findings = lint_extract_sidecar(payload, allow_legacy_rules=allow_legacy_rules)
            linted += 1
            if findings:
                failed_files += 1
                rows.extend({"file": rel, "code": f.code, "path": f.path, "message": f.message} for f in findings)

        if linted == 0:
            print(f"Error: no extract sidecars under {root} ({skipped} non-extract sidecars skipped).", file=sys.stderr)
            return 2

        summary = {
            "root": str(root),
            "linted": linted,
            "skipped": skipped,
            "files_with_findings": failed_files,
            "findings": rows,
        }
        if format == "json":
            print(json.dumps(summary, indent=2))
        else:
            for row in rows:
                print(f"ERROR {row['file']}: {row['code']} {row['path']}: {row['message']}")
            print(
                f"{linted} extract sidecar(s) linted, {failed_files} with findings ({len(rows)} finding(s)), "
                f"{skipped} non-extract sidecar(s) skipped"
            )
        return 1 if rows else 0
