#!/usr/bin/env python3
"""Audit supplementary CSVs and merge them into ordered Excel workbooks.

Run from the repository root::

    PYTHONPATH=. python tools/build_supplementary_xlsx.py
    PYTHONPATH=. python tools/build_supplementary_xlsx.py --report-only
    PYTHONPATH=. python tools/build_supplementary_xlsx.py --combined

Writes one workbook per manuscript figure under ``data/supplementary/xlsx/``
(e.g. ``Fig3_supplementary_tables.xlsx``) with ``Input_`` / ``Stats_`` sheets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print orphan/missing/duplicate report without writing Excel files",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write a single combined supplementary_tables.xlsx",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for per-figure workbooks (default: data/supplementary/xlsx)",
    )
    args = parser.parse_args()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from dlbcl.dlbcl_io import (
        audit_supplementary_tables,
        build_supplementary_tables_by_figure,
        build_supplementary_tables_xlsx,
        rel_path,
    )

    report = audit_supplementary_tables(REPO_ROOT)
    print(f"Supplementary CSVs: {report['n_csv']} files, {report['n_unique_stems']} unique stems")

    orphans = report["orphans_on_disk"]
    missing = report["missing_from_disk"]
    duplicates = report["duplicates"]

    if orphans:
        print(f"\nOrphans on disk (not in registry): {len(orphans)}")
        for stem in orphans:
            print(f"  - {stem}")
            for path in report["paths_by_stem"].get(stem, []):
                print(f"      {path}")
    else:
        print("\nOrphans on disk: none")

    if missing:
        print(f"\nRegistered but missing on disk: {len(missing)}")
        for stem in missing:
            print(f"  - {stem}")
    else:
        print("\nRegistered but missing on disk: none")

    if duplicates:
        print(f"\nDuplicate stems (same CSV in multiple folders): {len(duplicates)}")
        for stem, paths in duplicates.items():
            print(f"  - {stem}")
            for path in paths:
                print(f"      {path}")
    else:
        print("\nDuplicate stems: none")

    if args.report_only:
        return 0

    written = build_supplementary_tables_by_figure(REPO_ROOT, out_dir=args.out_dir)
    print(f"\nWrote {len(written)} per-figure workbook(s):")
    for path in written:
        print(f"  - {rel_path(path, REPO_ROOT)}")

    if args.combined:
        combined = build_supplementary_tables_xlsx(REPO_ROOT)
        print(f"  - {rel_path(combined, REPO_ROOT)} (combined)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
