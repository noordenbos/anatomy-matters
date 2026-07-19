#!/usr/bin/env python3
"""Patch notebook path prints and sanitize stored outputs for public release.

Run from the repository root::

    python tools/publish_notebooks.py
    python tools/publish_notebooks.py --check   # fail if private paths remain
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

DLBCL_IMPORT_NAMES = ("rel_path", "log_wrote", "log_saved")

# Exact source-line replacements (old, new).
SOURCE_LINE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ('print(f"AnnData: {ADATA_PATH}")', 'print(f"AnnData: {rel_path(ADATA_PATH, REPO_ROOT)}")'),
    ('print(f"Output:  {OUT_SVG}")', 'print(f"Output:  {rel_path(OUT_SVG, REPO_ROOT)}")'),
    ('print(f"Wrote {OUT_PNG}")', "log_wrote(OUT_PNG, REPO_ROOT)"),
    ('print(f"Wrote {OUT_SVG}")', "log_wrote(OUT_SVG, REPO_ROOT)"),
    ('print(f"Wrote {out_path}")', "log_wrote(out_path, REPO_ROOT)"),
    ('print(f"Wrote {FIG_4A}")', "log_wrote(FIG_4A, REPO_ROOT)"),
    ('print(f"Wrote {FIG_4A_LEGEND}")', "log_wrote(FIG_4A_LEGEND, REPO_ROOT)"),
    ('print(f"Wrote {FIG_6A}")', "log_wrote(FIG_6A, REPO_ROOT)"),
    ('print(f"Wrote {FIG_6A_LEGEND}")', "log_wrote(FIG_6A_LEGEND, REPO_ROOT)"),
    ('print(f"Wrote {FIG_6B}")', "log_wrote(FIG_6B, REPO_ROOT)"),
    ('print(f"Wrote {_fig4c}")', "log_wrote(_fig4c, REPO_ROOT)"),
    ('print(f"Wrote {_fig4d}")', "log_wrote(_fig4d, REPO_ROOT)"),
    ('print(f"Saved: {_ordered_csv}")', "log_saved(_ordered_csv, REPO_ROOT)"),
    ('print(f"Saved: {assoc_csv}")', "log_saved(assoc_csv, REPO_ROOT)"),
    ('print(f"Saved: {heatmap_png}")', "log_saved(heatmap_png, REPO_ROOT)"),
    ('print(f"Saved: {heatmap_pdf}")', "log_saved(heatmap_pdf, REPO_ROOT)"),
    ('print(f"Saved: {svg_file}")', "log_saved(svg_file, REPO_ROOT)"),
    ('print(f"Saved: {png_file}")', "log_saved(png_file, REPO_ROOT)"),
    ('print(f"Saved: {pdf_file}")', "log_saved(pdf_file, REPO_ROOT)"),
    (
        'print(f"Saved model bundle to {OUTDIR / \'cluster_assignment_elasticnet_model.joblib\'}")',
        "log_saved(OUTDIR / 'cluster_assignment_elasticnet_model.joblib', REPO_ROOT)",
    ),
    ('print(f"\\nSaved outputs to {OUTDIR.resolve()}")', 'print(f"\\nSaved outputs to {rel_path(OUTDIR, REPO_ROOT)}")'),
    ('print(f"\\nSaved clinical Cox outputs to {OUTDIR.resolve()}")', 'print(f"\\nSaved clinical Cox outputs to {rel_path(OUTDIR, REPO_ROOT)}")'),
    ('print(f"\\nSaved decomposition figures to {OUTDIR.resolve()}")', 'print(f"\\nSaved decomposition figures to {rel_path(OUTDIR, REPO_ROOT)}")'),
    ('print(f"\\nSaved ROC outputs to: {FIGDIR.resolve()}")', 'print(f"\\nSaved ROC outputs to: {rel_path(FIGDIR, REPO_ROOT)}")'),
    ('print(f"Saved SVG → {outpath.resolve()}")', 'print(f"Saved SVG → {rel_path(outpath, REPO_ROOT)}")'),
    ('print(f"Saved PNG → {outpath.resolve()}")', 'print(f"Saved PNG → {rel_path(outpath, REPO_ROOT)}")'),
    ('print(f"Saved at {figname_png}")', 'print(f"Saved at {rel_path(figname_png, REPO_ROOT)}")'),
    ('print(f"Saved SVG to: {svg_path}")', 'print(f"Saved SVG to: {rel_path(svg_path, REPO_ROOT)}")'),
    (
        'print(f"Saved {len(arch_df)} archetype assignments -> {ARCH_PATH}")',
        'print(f"Saved {len(arch_df)} archetype assignments -> {rel_path(ARCH_PATH, REPO_ROOT)}")',
    ),
    ('print(f"Outputs written to:\\n{OUTDIR}")', 'print(f"Outputs written to:\\n{rel_path(OUTDIR, REPO_ROOT)}")'),
    ('print(f"Saved to: {OUTDIR}")', 'print(f"Saved to: {rel_path(OUTDIR, REPO_ROOT)}")'),
    ('print(f"Saved:\\n{save_path}")', 'print(f"Saved:\\n{rel_path(save_path, REPO_ROOT)}")'),
    ('print(f"Saved:\\n{save_path}\\n{svg_path}")', 'print(f"Saved:\\n{rel_path(save_path, REPO_ROOT)}\\n{rel_path(svg_path, REPO_ROOT)}")'),
    ('print(f"Saved:\\n{png_path}\\n{svg_path}")', 'print(f"Saved:\\n{rel_path(png_path, REPO_ROOT)}\\n{rel_path(svg_path, REPO_ROOT)}")'),
    ('print(stem.with_suffix(".svg"))', 'print(rel_path(stem.with_suffix(".svg"), REPO_ROOT))'),
    ('print(raster_stem.with_suffix(".svg"))', 'print(rel_path(raster_stem.with_suffix(".svg"), REPO_ROOT))'),
    ('print(out_pdf)', 'print(rel_path(out_pdf, REPO_ROOT))'),
    ('print(out_path)', 'print(rel_path(out_path, REPO_ROOT))'),
    ('print(OUTDIR)', 'print(rel_path(OUTDIR, REPO_ROOT))'),
    ('print(GSEA_OUTDIR)', 'print(rel_path(GSEA_OUTDIR, REPO_ROOT))'),
    ('print(png_file)', 'print(rel_path(png_file, REPO_ROOT))'),
    ('print(svg_file)', 'print(rel_path(svg_file, REPO_ROOT))'),
    ('print(pdf_file)', 'print(rel_path(pdf_file, REPO_ROOT))'),
    (
        'print(os.path.join(OUTDIR, "HLAABC_HLADR_global_stats.csv"))',
        'print(rel_path(OUTDIR / "HLAABC_HLADR_global_stats.csv", REPO_ROOT))',
    ),
    (
        'print(" ".join(cmd))',
        'print(" ".join(rel_path(Path(a), REPO_ROOT) if not str(a).startswith("-") else str(a) for a in cmd))',
    ),
    (
        'print("Discovery table:", DISCOVERY_TABLE_CSV)',
        'print("Discovery table:", rel_path(DISCOVERY_TABLE_CSV, REPO_ROOT))',
    ),
)

SAVED_OUTDIR_LINES = (
    'print(f"Saved: {OUTDIR / \'location_association_results.csv\'}")',
    'print(f"Saved: {OUTDIR / \'location_pairwise_results.csv\'}")',
    'print(f"Saved: {OUTDIR / \'archetype_association_results.csv\'}")',
    'print(f"Saved: {OUTDIR / \'spatial_protein_pairwise_results.csv\'}")',
    'print(f"Saved: {OUTDIR / \'ecotyper_b_state_by_archetype_chisq_stats.csv\'}")',
)

PRIVATE_PATH_PATTERNS = (
    re.compile(re.escape(str(REPO_ROOT.resolve()))),
    re.compile(r"/Users/[^/\s\"']+/code/DLBCL_location_2026"),
    re.compile(r"/Users/[^/\s\"']+/"),
)


def _split_import_names(text: str) -> set[str]:
    names: set[str] = set()
    for part in re.split(r"[,\n]", text):
        part = part.strip()
        if part:
            names.add(part)
    return names


def _find_dlbcl_import_span(src: str) -> tuple[int, int, set[str]] | None:
    """Return (start, end, imported_names) for the first ``dlbcl_io`` import."""
    match = re.search(r"^from dlbcl\.dlbcl_io import ", src, flags=re.MULTILINE)
    if not match:
        return None
    start = match.start()
    index = match.end()
    while index < len(src) and src[index] in " \t":
        index += 1
    if index < len(src) and src[index] == "(":
        depth = 0
        for j in range(index, len(src)):
            char = src[j]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return start, j + 1, _split_import_names(src[index + 1 : j])
        return None
    end = src.find("\n", index)
    if end == -1:
        end = len(src)
    return start, end, _split_import_names(src[index:end])


def _add_dlbcl_imports(src: str) -> str:
    needs = [name for name in DLBCL_IMPORT_NAMES if re.search(rf"\b{name}\b", src)]
    if not needs:
        return src
    span = _find_dlbcl_import_span(src)
    if span is None:
        return src
    start, end, existing = span
    missing = [name for name in needs if name not in existing]
    if not missing:
        return src
    block = src[start:end]
    if "(" in block:
        close = block.rfind(")")
        indent_match = re.search(r"\n(\s+)\S", block)
        indent = indent_match.group(1) if indent_match else "    "
        addition = ",\n".join(f"{indent}{name}" for name in missing)
        before_close = block[:close].rstrip().rstrip(",")
        if before_close.endswith("("):
            new_block = f"{before_close}\n{addition}\n{block[close:]}"
        else:
            new_block = f"{before_close},\n{addition}\n{block[close:]}"
    else:
        new_block = block.rstrip() + ", " + ", ".join(missing)
    return src[:start] + new_block + src[end:]


def _patch_nb1d_imports(src: str) -> str:
    if "from dlbcl.dlbcl_io import configure_notebook" in src and "rel_path" not in src:
        return src.replace(
            "from dlbcl.dlbcl_io import configure_notebook",
            "from dlbcl.dlbcl_io import configure_notebook, log_wrote, rel_path",
        )
    return src


def patch_notebook_source(src: str) -> str:
    out = _add_dlbcl_imports(src)
    out = _patch_nb1d_imports(out)
    for old, new in SOURCE_LINE_REPLACEMENTS:
        out = out.replace(old, new)
    for line in SAVED_OUTDIR_LINES:
        if line in out:
            inner = line[len('print(f"Saved: {') : -len('}")')]
            out = out.replace(line, f"log_saved({inner}, REPO_ROOT)")
    return out


def sanitize_text(text: str) -> str:
    if not text:
        return text
    root = str(REPO_ROOT.resolve())
    text = text.replace(root, ".")
    text = re.sub(r"/Users/[^/\s\"']+/code/DLBCL_location_2026", ".", text)
    text = re.sub(r"/Users/[^/\s\"']+/", "~/", text)
    text = re.sub(
        r"[^\s\"']*?/\.venv/lib/[^/\s\"']+/site-packages/",
        ".venv/lib/python/site-packages/",
        text,
    )
    return text


def _sanitize_output_obj(obj: dict) -> None:
    if obj.get("output_type") == "stream":
        obj.setdefault("name", "stdout")
        if "text" in obj:
            if isinstance(obj["text"], list):
                obj["text"] = [sanitize_text(part) for part in obj["text"]]
            else:
                obj["text"] = sanitize_text(obj["text"])
    elif obj.get("output_type") == "error":
        for key in ("traceback", "ename", "evalue"):
            if key in obj:
                if isinstance(obj[key], list):
                    obj[key] = [sanitize_text(part) for part in obj[key]]
                else:
                    obj[key] = sanitize_text(obj[key])
    elif obj.get("output_type") in {"execute_result", "display_data"}:
        obj.setdefault("metadata", {})
        data = obj.get("data", {})
        for mime in ("text/plain", "text/html", "application/javascript"):
            if mime not in data:
                continue
            if isinstance(data[mime], list):
                data[mime] = [sanitize_text(part) for part in data[mime]]
            else:
                data[mime] = sanitize_text(data[mime])


def sanitize_notebook(nb: dict) -> int:
    changed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            before = json.dumps(output, sort_keys=True)
            _sanitize_output_obj(output)
            if json.dumps(output, sort_keys=True) != before:
                changed += 1
    return changed


def patch_notebook(nb: dict) -> bool:
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        patched = patch_notebook_source(src)
        if patched != src:
            cell["source"] = [line if line.endswith("\n") else line + "\n" for line in patched.splitlines(keepends=False)]
            if cell["source"] and not src.endswith("\n") and cell["source"][-1].endswith("\n"):
                cell["source"][-1] = cell["source"][-1].rstrip("\n")
            changed = True
    return changed


def contains_private_paths(text: str) -> bool:
    return any(pattern.search(text) for pattern in PRIVATE_PATH_PATTERNS)


def _figure_notebook_paths() -> list[Path]:
    names = sorted(NOTEBOOKS_DIR.glob("fig*.ipynb")) + sorted(
        NOTEBOOKS_DIR.glob("supplemental_fig*.ipynb")
    )
    return names


def process_notebooks(*, check_only: bool = False) -> int:
    failures: list[str] = []
    for nb_path in _figure_notebook_paths():
        raw = nb_path.read_text(encoding="utf-8")
        if check_only:
            if contains_private_paths(raw):
                failures.append(nb_path.name)
            continue

        nb = json.loads(raw)
        patched = patch_notebook(nb)
        sanitized = sanitize_notebook(nb)
        if patched or sanitized:
            nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"{nb_path.name}: patched={patched} sanitized_outputs={sanitized}")
        elif contains_private_paths(raw):
            failures.append(nb_path.name)

    if failures:
        print("Private paths still present in:", ", ".join(failures), file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only verify notebooks contain no private paths")
    args = parser.parse_args()
    return process_notebooks(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
