#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/dissertation_model_selection.py"

usage() {
  cat <<'EOF'
Usage: run_dissertation_model_selection_all_he_she_odds.sh [extra python args...]

Runs dissertation_model_selection.py once for each he_she_odds_results__*.csv
file in the repository root.

Notes:
  - Extra arguments are forwarded to the Python script.
  - Do not pass a positional results CSV; the wrapper supplies one for each file.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --help|-h)
      usage
      exit 0
      ;;
    he_she_odds_results__*.csv|*/he_she_odds_results__*.csv)
      printf 'Error: do not pass a results CSV to the wrapper: %s\n' "$arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  printf 'Error: Python script not found: %s\n' "$PYTHON_SCRIPT" >&2
  exit 1
fi

shopt -s nullglob
csv_files=("${SCRIPT_DIR}"/he_she_odds_results__*.csv)
shopt -u nullglob

if [[ ${#csv_files[@]} -eq 0 ]]; then
  printf 'Error: no he_she_odds_results__*.csv files found in %s\n' "$SCRIPT_DIR" >&2
  exit 1
fi

for csv_file in "${csv_files[@]}"; do
  printf 'Running selection for: %s\n' "$(basename "$csv_file")"
  uv run python "$PYTHON_SCRIPT" "$csv_file" "$@"
done
