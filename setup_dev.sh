#!/usr/bin/env bash
# Bootstrap a local checkout for figure reproduction.
#
# Recommended (Zenodo download when published):
#   ./setup_dev.sh
#
# Workaround until Zenodo is live — point at a local AnnData file (symlink, no copy):
#   ./setup_dev.sh --adata /path/to/DLBCL_location_2026.h5ad
#   ./setup_dev.sh /path/to/DLBCL_location_2026.h5ad          # same as --adata
#
# Optional: run all figure notebooks after setup (~10–20 minutes typical):
#   ./setup_dev.sh --execute-figures --adata /path/to/DLBCL_location_2026.h5ad
#
# Override Zenodo record without editing this file:
#   ZENODO_RECORD_ID=12345678 ./setup_dev.sh

set -euo pipefail

# Set this when the AnnData bundle is published on Zenodo (digits only).
# Leave empty to require --adata / local path until then.
DEFAULT_ZENODO_RECORD_ID=""

ADATA_BASENAME="DLBCL_location_2026.h5ad"
EXECUTE_FIGURES=0
ADATA_SRC=""
ARGS=()

usage() {
  cat <<'EOF'
Usage: ./setup_dev.sh [options] [/path/to/DLBCL_location_2026.h5ad]

Options:
  --adata PATH          Symlink PATH into data/DLBCL_location_2026.h5ad
  --execute-figures     Run all figure notebooks after setup
  -h, --help            Show this help

If no local AnnData is given, the script downloads from Zenodo when
ZENODO_RECORD_ID is set (env or DEFAULT_ZENODO_RECORD_ID in this file).
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --execute-figures|--execute-notebooks)
      EXECUTE_FIGURES=1
      ;;
    --adata)
      # next arg handled below via ARGS sentinel
      ARGS+=("--adata")
      ;;
    *)
      ARGS+=("$arg")
      ;;
  esac
done

# Parse --adata VALUE or a bare positional path.
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
  arg="${ARGS[$i]}"
  if [[ "$arg" == "--adata" ]]; then
    i=$((i + 1))
    if [[ $i -ge ${#ARGS[@]} ]]; then
      echo "ERROR: --adata requires a path" >&2
      exit 1
    fi
    ADATA_SRC="${ARGS[$i]}"
  elif [[ -z "$ADATA_SRC" && "$arg" != -* ]]; then
    ADATA_SRC="$arg"
  else
    echo "ERROR: unrecognized argument: $arg" >&2
    usage >&2
    exit 1
  fi
  i=$((i + 1))
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.10+ and retry." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment (.venv) ..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

ADATA_LINK="data/${ADATA_BASENAME}"
mkdir -p data

link_or_copy_adata() {
  local src="$1"
  if [[ ! -f "$src" ]]; then
    echo "ERROR: AnnData file not found: $src" >&2
    exit 1
  fi
  src="$(cd "$(dirname "$src")" && pwd)/$(basename "$src")"
  if [[ -e "$ADATA_LINK" || -L "$ADATA_LINK" ]]; then
    echo "AnnData already present at $ADATA_LINK (leaving as-is)"
    return 0
  fi
  ln -s "$src" "$ADATA_LINK"
  echo "Symlinked $ADATA_LINK -> $src"
}

download_adata_from_zenodo() {
  local record_id="$1"
  local url="https://zenodo.org/records/${record_id}/files/${ADATA_BASENAME}?download=1"
  echo "Downloading AnnData from Zenodo (record ${record_id}) ..."
  echo "  $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar -o "$ADATA_LINK" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$ADATA_LINK" "$url"
  else
    echo "ERROR: need curl or wget to download from Zenodo." >&2
    exit 1
  fi
  echo "Saved $ADATA_LINK"
}

ZENODO_RECORD_ID="${ZENODO_RECORD_ID:-$DEFAULT_ZENODO_RECORD_ID}"

if [[ -n "$ADATA_SRC" ]]; then
  link_or_copy_adata "$ADATA_SRC"
elif [[ -e "$ADATA_LINK" || -L "$ADATA_LINK" ]]; then
  echo "AnnData already present at $ADATA_LINK"
elif [[ -n "$ZENODO_RECORD_ID" ]]; then
  download_adata_from_zenodo "$ZENODO_RECORD_ID"
else
  cat <<EOF

AnnData is not installed yet, and Zenodo download is not enabled
(DEFAULT_ZENODO_RECORD_ID is empty / ZENODO_RECORD_ID unset).

Workaround — symlink a local copy (no file copy; ~0.6–2 GB):

  ./setup_dev.sh --adata /path/to/${ADATA_BASENAME}

When the Zenodo record is published, either set DEFAULT_ZENODO_RECORD_ID
in setup_dev.sh or run:

  ZENODO_RECORD_ID=<record_id> ./setup_dev.sh

EOF
fi

adata_ready() {
  [[ -e "$ADATA_LINK" || -L "$ADATA_LINK" ]]
}

run_all_figure_notebooks() {
  if ! adata_ready; then
    echo "ERROR: cannot run notebooks without $ADATA_LINK" >&2
    return 1
  fi
  echo "Executing figure notebooks (most finish in under ~5 minutes each; full suite often ~10–20 minutes) ..."
  PYTHONPATH=. python tools/run_all_notebooks.py
}

launch_jupyter() {
  if ! adata_ready; then
    echo "ERROR: cannot open notebooks without $ADATA_LINK" >&2
    return 1
  fi
  echo "Starting Jupyter in notebooks/ (Ctrl+C to stop the server) ..."
  (cd notebooks && jupyter notebook)
}

print_completion() {
  echo
  echo "Setup complete."
  echo
  echo "Next time, first activate the environment:"
  echo "  source .venv/bin/activate"
  echo
  echo "Then either:"
  echo "  (1) Explore notebooks in the browser:"
  echo "        cd notebooks && jupyter notebook"
  echo "  (2) Re-run all notebooks (figures + tables):"
  echo "        PYTHONPATH=. python tools/run_all_notebooks.py"
  echo
  if adata_ready; then
    echo "AnnData: $ADATA_LINK"
  else
    echo "AnnData: missing — re-run with --adata PATH (see above)"
  fi
}

if [[ "$EXECUTE_FIGURES" == "1" ]]; then
  run_all_figure_notebooks
elif [[ -t 0 ]] && adata_ready; then
  echo
  echo "What next?"
  echo "  1) Explore notebooks interactively in the Jupyter browser"
  echo "  2) Run all notebooks and recreate figures and data tables locally"
  echo "  3) Exit"
  echo
  read -r -p "Enter choice [1/2/3]: " choice
  case "${choice// /}" in
    1)
      launch_jupyter || true
      ;;
    2)
      run_all_figure_notebooks || true
      ;;
    3|"")
      ;;
    *)
      echo "Unrecognized choice ('$choice'); continuing."
      ;;
  esac
fi

print_completion
