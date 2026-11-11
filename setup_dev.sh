#!/usr/bin/env bash
# Install the paired code/data release inside this checkout.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
DATA_ARGS=()
EXECUTE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --adata) DATA_ARGS+=(--study-source "$2"); shift 2 ;;
    --data-dir) DATA_ARGS+=(--source-dir "$2"); shift 2 ;;
    --execute-figures|--execute-notebooks) EXECUTE=1; shift ;;
    -h|--help) echo 'Usage: ./setup_dev.sh [--data-dir DIR] [--adata FILE] [--execute-figures]'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints.txt -r requirements.txt
python tools/fetch_release_data.py "${DATA_ARGS[@]}"
if [[ "$EXECUTE" == 1 ]]; then
  PYTHONPATH=. python tools/run_all_notebooks.py
fi
echo 'Setup complete. Activate with: source .venv/bin/activate'
echo 'Open notebooks with: cd notebooks && jupyter notebook'
echo 'Recreate figures with: PYTHONPATH=. python tools/run_all_notebooks.py'
