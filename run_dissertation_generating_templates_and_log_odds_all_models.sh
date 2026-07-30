#!/bin/bash --login
#SBATCH -p gpuA40GB
#SBATCH -G 4
#SBATCH -t 1-0
#SBATCH -n 8

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)."

set -euo pipefail

# Slurm may stage the batch script in /var/spool/slurmd, so BASH_SOURCE[0]
# does not necessarily refer to the submission directory.  Use Slurm's
# submission directory for both the working directory and project files.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"

PYTHON_SCRIPT="${SUBMIT_DIR}/dissertation_generating_templates_and_log_odds.py"
MODEL_LIST="${SUBMIT_DIR}/model_list.txt"

usage() {
  cat <<'EOF'
Usage: run_dissertation_generating_templates_and_log_odds_all_models.sh [extra python args...]

Runs dissertation_generating_templates_and_log_odds.py once for each model listed in model_list.txt.

Notes:
  - Blank lines and lines starting with '#' are ignored.
  - Extra arguments are forwarded to the Python script.
  - Do not pass --model-tag or --output; the wrapper sets --model-tag per model and
    relies on the Python script's default per-model output naming.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --model-tag|--model-tag=*|--output|--output=*)
      printf 'Error: %s must not be passed to the wrapper.\n' "$arg" >&2
      usage >&2
      exit 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
  esac
done

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  printf 'Error: Python script not found: %s\n' "$PYTHON_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$MODEL_LIST" ]]; then
  printf 'Error: Model list not found: %s\n' "$MODEL_LIST" >&2
  exit 1
fi

while IFS= read -r model_tag || [[ -n "$model_tag" ]]; do
  model_tag="${model_tag#"${model_tag%%[![:space:]]*}"}"
  model_tag="${model_tag%"${model_tag##*[![:space:]]}"}"

  if [[ -z "$model_tag" || "$model_tag" == \#* ]]; then
    continue
  fi

  printf 'Running model: %s\n' "$model_tag"
  uv run python "$PYTHON_SCRIPT" --model-tag "$model_tag" "$@"
done < "$MODEL_LIST"
