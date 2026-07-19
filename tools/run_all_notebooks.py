#!/usr/bin/env python3
"""Execute all figure notebooks and save them with inline outputs.

Run from the repository root::

    PYTHONPATH=. python tools/run_all_notebooks.py
    PYTHONPATH=. python tools/run_all_notebooks.py --only fig1 fig6
    PYTHONPATH=. python tools/run_all_notebooks.py --dry-run

Skips ``supplemental_fig1`` when ``Rscript`` is unavailable (oncoprint panel).
After each successful run, the notebook file is rewritten with cell outputs.

By default, tSNE panels write rasterized exports only (faster, smaller files).
Internal: pass ``--write-vector-tsne`` for full vector + ``*_raster`` dual export.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

FIGURE_NOTEBOOKS: tuple[str, ...] = (
    "fig1.ipynb",
    "fig2.ipynb",
    "fig3.ipynb",
    "fig4.ipynb",
    "fig5.ipynb",
    "fig6.ipynb",
    "supplemental_fig1.ipynb",
    "supplemental_fig2.ipynb",
    "supplemental_fig3.ipynb",
    "supplemental_fig4.ipynb",
    "supplemental_fig5.ipynb",
)

DEFAULT_TIMEOUT_SEC = 1800
TIMEOUT_SEC: dict[str, int] = {
    "fig2": 7200,
    "fig3": 7200,
    "fig5": 3600,
    "supplemental_fig2": 3600,
    "supplemental_fig3": 3600,
    "supplemental_fig5": 3600,
}


def _preflight(selected: list[str]) -> list[str]:
    errors: list[str] = []
    data_dir = Path(os.environ.get("DLBCL_DATA_DIR", REPO_ROOT / "data"))
    adata = data_dir / "DLBCL_location_2026.h5ad"
    if not adata.exists():
        errors.append(f"Missing AnnData bundle: {adata}")
    for name in selected:
        if not (NOTEBOOKS_DIR / name).exists():
            errors.append(f"Missing notebook: {name}")
    return errors


class _SuppressKernelTcpWarning:
    """Drop the noisy IPKernelApp TCP-without-encryption line from stderr."""

    _NEEDLE = "Kernel is running over TCP without encryption"

    def __init__(self, stream):
        self._stream = stream
        self._buf = ""

    def write(self, data: str) -> int:
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if self._NEEDLE not in line:
                self._stream.write(line + "\n")
        return len(data)

    def flush(self) -> None:
        if self._buf and self._NEEDLE not in self._buf:
            self._stream.write(self._buf)
        self._buf = ""
        self._stream.flush()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def execute_notebook_inprocess(nb_path: Path, *, timeout: int) -> float:
    """Execute one notebook in this process (used by the per-notebook worker)."""
    logging.getLogger("IPKernelApp").setLevel(logging.ERROR)
    nb = nbformat.read(nb_path, as_version=4)
    km = KernelManager(kernel_name="python3")
    # Prefer IPv4 loopback; avoids some Windows ZMQ / interface discovery issues.
    km.ip = "127.0.0.1"
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        km=km,
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
        store_widget_state=False,
        startup_timeout=120,
    )
    # Subprocess kernel ignores parent log levels; also lower its own if supported.
    client.extra_arguments = ["--IPKernelApp.log_level=50"]
    t0 = time.time()
    old_err = sys.stderr
    sys.stderr = _SuppressKernelTcpWarning(old_err)
    try:
        client.execute()
    finally:
        sys.stderr.flush()
        sys.stderr = old_err
        try:
            if client.km is not None and client.km.is_alive():
                client.km.shutdown_kernel(now=True)
        except Exception:
            pass
    nbformat.write(nb, nb_path)
    return time.time() - t0


def execute_notebook(nb_path: Path, *, timeout: int) -> float:
    """Run one notebook in an isolated child process.

    Isolation matters on Windows: Jupyter/libzmq can abort after fig1 when the
    parent still holds large notebook outputs in memory, which then looks like a
    hang until the long fig2 timeout. A fresh process releases that RAM and
    surfaces a hard crash immediately instead of waiting out the timeout.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    env.setdefault("DLBCL_TSNE_RASTER_ONLY", "1")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-execute",
        str(nb_path),
        "--worker-timeout",
        str(timeout),
    ]
    t0 = time.time()
    # Hard ceiling slightly above the notebook timeout so a dead kernel cannot
    # stall the suite for the full fig2 window without a clear failure.
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=timeout + 180,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"worker timed out after {timeout + 180}s while executing {nb_path.name}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker exited with code {proc.returncode} while executing {nb_path.name}"
        )
    return time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="FIG_ID",
        help="Execute only these figure ids (e.g. fig1 supplemental_fig3)",
    )
    parser.add_argument(
        "--skip-r",
        action="store_true",
        help="Skip supplemental_fig1 even when Rscript is available",
    )
    parser.add_argument(
        "--write-vector-tsne",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: full vector + *_raster tSNE exports
    )
    parser.add_argument(
        "--worker-execute",
        type=Path,
        help=argparse.SUPPRESS,  # internal: execute a single notebook in-process
    )
    parser.add_argument(
        "--worker-timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.worker_execute is not None:
        nb_path = args.worker_execute
        if not nb_path.is_absolute():
            nb_path = (REPO_ROOT / nb_path).resolve()
        try:
            elapsed = execute_notebook_inprocess(nb_path, timeout=args.worker_timeout)
        except CellExecutionError as exc:
            print(f"FAILED: {nb_path.name}\n{exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"FAILED: {nb_path.name}: {exc}", file=sys.stderr)
            return 1
        print(f"WORKER_OK {nb_path.name} ({elapsed / 60:.1f} min)", flush=True)
        return 0

    if args.write_vector_tsne:
        os.environ["DLBCL_TSNE_WRITE_VECTOR"] = "1"
        os.environ.pop("DLBCL_TSNE_RASTER_ONLY", None)
    else:
        os.environ.setdefault("DLBCL_TSNE_RASTER_ONLY", "1")

    id_to_name = {name.replace(".ipynb", ""): name for name in FIGURE_NOTEBOOKS}
    if args.only:
        selected: list[str] = []
        for token in args.only:
            key = token.replace("notebooks/", "").replace(".ipynb", "")
            if key not in id_to_name:
                print(f"ERROR: unknown figure id: {token}", file=sys.stderr)
                return 1
            selected.append(id_to_name[key])
    else:
        selected = list(FIGURE_NOTEBOOKS)

    errors = _preflight(selected)
    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    if not shutil.which("Rscript"):
        if "supplemental_fig1.ipynb" in selected:
            print("WARN: Rscript not found — skipping supplemental_fig1.ipynb")
            selected = [n for n in selected if n != "supplemental_fig1.ipynb"]
    elif args.skip_r and "supplemental_fig1.ipynb" in selected:
        selected = [n for n in selected if n != "supplemental_fig1.ipynb"]

    print(f"Executing {len(selected)} figure notebook(s):")
    for name in selected:
        print(f"  - {name}")

    if args.dry_run:
        return 0

    total_t0 = time.time()
    failures: list[str] = []
    for i, name in enumerate(selected, start=1):
        nb_path = NOTEBOOKS_DIR / name
        fig_id = name.replace(".ipynb", "")
        timeout = TIMEOUT_SEC.get(fig_id, DEFAULT_TIMEOUT_SEC)
        print(f"\n[{i}/{len(selected)}] Executing {name} ...", flush=True)
        try:
            elapsed = execute_notebook(nb_path, timeout=timeout)
        except CellExecutionError as exc:
            print(f"\nFAILED: {name}\n{exc}", file=sys.stderr)
            failures.append(name)
            continue
        except Exception as exc:
            print(f"\nFAILED: {name}: {exc}", file=sys.stderr)
            failures.append(name)
            continue
        print(f"OK {name} ({elapsed / 60:.1f} min)", flush=True)

    total_min = (time.time() - total_t0) / 60
    if failures:
        print(f"\n{len(failures)} notebook(s) failed: {', '.join(failures)}", file=sys.stderr)
        print(f"Finished in {total_min:.1f} min", file=sys.stderr)
        return 1

    print(f"\nAll {len(selected)} figure notebooks finished in {total_min:.1f} min")
    print("Notebook files updated with inline outputs.")
    try:
        from dlbcl.dlbcl_io import build_supplementary_tables_by_figure, rel_path

        written = build_supplementary_tables_by_figure(REPO_ROOT)
        if written:
            print(f"Merged supplementary tables into {len(written)} Excel workbook(s):")
            for path in written:
                print(f"  - {rel_path(path, REPO_ROOT)}")
    except Exception as exc:
        print(f"WARN: supplementary Excel merge skipped ({exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
