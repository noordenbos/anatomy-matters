# Anatomical Origin Defines Tumor–Immune Archetypes in Diffuse Large B-cell Lymphoma

Code and notebooks to reproduce the figures in our manuscript (preprint; journal submission pending).

Notebooks are committed **with inline outputs**, so you can browse results without re-running.
We do invite reviewers/editors/public to rerun the notebooks as a fully reproducible pipeline, this process is simplified to 2 simple steps to lower participation threshold.
There are twelve independent Jupyter notebooks (`fig1`–`fig6`, `table1`, `supplemental_fig1`–`supplemental_fig5`) which read the same single AnnData file (~600MB), that is automatically pulled from zenodo. 

---

## Recommended setup (use the script)

| Platform | Setup script |
|----------|----------------|
| macOS / Linux / [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) | `./setup_dev.sh` |
| Windows PowerShell | `.\setup_dev.ps1` |

### 1. Get the code

```bash
git clone https://github.com/noordenbos/anatomy-matters.git
cd anatomy-matters
```

No Git? See [First time with Python / Git](#first-time-with-python--git) (ZIP download workaround).

### 2. Run the setup script

**macOS / Linux / WSL**

```bash
chmod +x setup_dev.sh
./setup_dev.sh
```

**Windows PowerShell**

```powershell
# If blocked once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\setup_dev.ps1
```

This downloads **only** the AnnData (~600 MB) from Zenodo record [21440631](https://zenodo.org/records/21440631) into `data/DLBCL_location_2026.h5ad`. It does **not** download the raw IMC archive (`IMC.7z`).

To use a local AnnData instead: `./setup_dev.sh --adata /path/to/DLBCL_location_2026.h5ad` (or `.\setup_dev.ps1 -Adata ...`).

Optional: execute every figure notebook after setup (most notebooks finish in under ~5 minutes; full suite often ~10–20 minutes):

```bash
./setup_dev.sh --execute-figures
```

```powershell
.\setup_dev.ps1 -ExecuteFigures
```

The script creates `.venv`, installs `requirements.txt`, places `data/DLBCL_location_2026.h5ad`, then offers a short menu (Jupyter / re-run all / exit) and prints the commands to use next time.

Everything the setup creates stays inside this folder (virtualenv, AnnData link or copy, regenerated figures). Delete the folder to remove the local install completely — nothing is written to system Python, global Jupyter, or other paths outside the checkout. If you pointed `-Adata` / `--adata` at a file elsewhere, that original file is left untouched.

### 3. Later sessions

First activate the environment, then choose Jupyter or a full re-run:

```bash
source .venv/bin/activate
cd notebooks && jupyter notebook          # (1) interactive
# or from repo root:
PYTHONPATH=. python tools/run_all_notebooks.py   # (2) recreate figures + tables
```

```powershell
.\.venv\Scripts\Activate.ps1
cd notebooks; jupyter notebook
# or from repo root:
$env:PYTHONPATH = "$PWD"; python tools/run_all_notebooks.py
```

`supplemental_fig1` (oncoprint) needs R — see [Fig S1A (R)](#fig-s1a-r-oncoprint-only). It is skipped automatically if `Rscript` is missing.

---

## First time with Python / Git?

You only need a terminal, **Python 3.10+**, and either **Git** or a ZIP download.

| Tool | macOS | Windows |
|------|--------|---------|
| Terminal | Terminal.app | PowerShell, or [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) |
| Python | [python.org](https://www.python.org/downloads/) or `brew install python` | [python.org](https://www.python.org/downloads/) (tick **Add python.exe to PATH**) or WSL: `sudo apt install python3 python3-venv` |
| Git | `xcode-select --install` or [git-scm.com](https://git-scm.com/) | [git-scm.com](https://git-scm.com/) or WSL: `sudo apt install git` |
| Setup | `./setup_dev.sh` | `.\setup_dev.ps1` (or `./setup_dev.sh` inside WSL) |

Check:

```bash
python3 --version    # should be 3.10 or newer
git --version        # optional if you use the ZIP workaround below
```

### ZIP download (no Git)

1. Open the GitHub repo.
2. **Code → Download ZIP**.
3. Unzip, then in a terminal:

**macOS / Linux / WSL**

```bash
cd /path/to/anatomy-matters-main   # folder name may include -main
chmod +x setup_dev.sh
./setup_dev.sh --adata /path/to/DLBCL_location_2026.h5ad
```

**Windows PowerShell**

```powershell
cd C:\path\to\anatomy-matters-main
.\setup_dev.ps1 -Adata C:\path\to\DLBCL_location_2026.h5ad
```

ZIP checkouts are fine for reproduction; you just will not receive `git pull` updates unless you switch to a clone later.

### Common snags

- **`python3` / `python` not found** — install Python 3.10+, reopen the terminal; on Windows tick **Add python.exe to PATH**.
- **`permission denied: ./setup_dev.sh`** — run `chmod +x setup_dev.sh`.
- **PowerShell: running scripts is disabled** — `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
- **Symlink fails on Windows** — enable Developer Mode, or let `setup_dev.ps1` fall back to copying the h5ad.
- **AnnData missing** — until Zenodo is online pass `--adata` / `-Adata` with the full path to `DLBCL_location_2026.h5ad`.
- **Windows: `Assertion failed: error not defined ... libzmq ... ip.cpp` while batch-running notebooks** — usually memory pressure when starting the next kernel (fig2 is the heaviest early notebook). Close other apps, aim for ~16 GB RAM free, `git pull` for the latest runner (isolates each notebook in a subprocess), then resume with e.g. `$env:PYTHONPATH = "$PWD"; python tools/run_all_notebooks.py --only fig2 fig3 fig4 fig5 fig6 supplemental_fig2 supplemental_fig3 supplemental_fig4 supplemental_fig5`.

---

## Figure notebooks

| Notebook | Manuscript | Contents (summary) |
|----------|------------|-------------------|
| `fig1.ipynb` | Fig 1 | Classifier composition pies, location KM |
| `fig2.ipynb` | Fig 2 | tSNE landscape, archetype heatmap/pies, archetype KM |
| `fig3.ipynb` | Fig 3 | Spatial segmentation, transcriptome programs, HLA |
| `fig4.ipynb` | Fig 4 | Discovery integration (circos, associations, enrichment) |
| `fig5.ipynb` | Fig 5 | Elastic-net classifier, validation survival/Cox |
| `fig6.ipynb` | Fig 6 | Validation integration |
| `table1.ipynb` | Table 1 | Clinical characteristics (discovery + validation) |
| `supplemental_fig1.ipynb` | Fig S1 | Genomic oncoprint (needs R) |
| `supplemental_fig2.ipynb` | Fig S2 | Modality matrix, QC tSNE, compartment tSNEs, protein heatmaps |
| `supplemental_fig3.ipynb` | Fig S3 | Gene–protein validation, PCA, HLA panels, classifier tSNEs |
| `supplemental_fig4.ipynb` | Fig S4 | Discovery/validation Cox supplements |
| `supplemental_fig5.ipynb` | Fig S5 | Validation location KM, EcoTyper, module scores, GSEA |

Cross-figure panels that share computation are markdown pointers (for example Fig S3J → `fig5.ipynb`).

```
notebooks/          # One notebook per manuscript figure
dlbcl/              # Shared helpers imported by the notebooks
tools/              # Batch runner (run_all_notebooks.py)
setup_dev.sh        # Setup (macOS / Linux / WSL)
setup_dev.ps1       # Setup (Windows PowerShell)

requirements.txt    # Python dependencies
data/               # DLBCL_location_2026.h5ad (not committed)
figures/            # SVG/PNG outputs (created when you run notebooks)
```

---

## Data access

Single file: **`data/DLBCL_location_2026.h5ad`**. It contains single-cell protein data, patient-level expression/metadata, spatial masks, OMIQ tSNE coordinates, genomic profiling tables, and the embedded validation cohort (`uns['validation_cohort']`, aliases `V1`…).

Tumor–immune archetype labels (`tumorimmune_archetype` / `tumorimmune_archetype_id`) and genomic classifier columns live in `adata.uns` — no separate CSV files are required. Override the data root with `DLBCL_DATA_DIR` if needed.

When notebooks are re-run, plot source tables export to `data/supplementary/{fig_id}/` (gitignored).

_Patient-level restricted data are not hosted in this repository._

---

## Fig S1A (R oncoprint only)

Install [R](https://cran.r-project.org/) (≥ 4.2), then:

```r
install.packages("svglite")
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("ComplexHeatmap")
```

Confirm `Rscript` is on your `PATH` (`which Rscript`).

---

## Manual setup reference

Prefer [`./setup_dev.sh`](#recommended-setup-use-the-script) or [`.\setup_dev.ps1`](#recommended-setup-use-the-script). These steps are equivalent if you set things up by hand.

<details>
<summary>macOS / Linux (click to expand)</summary>

```bash
git clone https://github.com/noordenbos/anatomy-matters.git
cd anatomy-matters
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p data
# AnnData only (~600 MB); do not download IMC.7z from the same Zenodo record
curl -fL -o data/DLBCL_location_2026.h5ad \
  "https://zenodo.org/records/21440631/files/DLBCL_location_2026.publish.h5ad?download=1"
```

</details>

<details>
<summary>Windows PowerShell (click to expand)</summary>

Prefer `.\setup_dev.ps1 -Adata C:\path\to\DLBCL_location_2026.h5ad`. Manual equivalent:

```powershell
git clone https://github.com/noordenbos/anatomy-matters.git
cd anatomy-matters
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

New-Item -ItemType Directory -Force -Path data | Out-Null
# Symlink (may require Developer Mode); otherwise Copy-Item the h5ad into data\
New-Item -ItemType SymbolicLink -Path "data\DLBCL_location_2026.h5ad" `
  -Target "C:\path\to\DLBCL_location_2026.h5ad"
```

If script activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

</details>

---

## Citation

_Preprint citation will be added when available._

## License

See [LICENSE](LICENSE).
