#!/bin/bash --login
#SBATCH -p gpuA40GB
#SBATCH -G 2
#SBATCH -n 24
#SBATCH -t 2-0

set -euo pipefail

# Regenerate the random-slopes, profession-embedding, and knowledge/perception
# reports from the modelling CSVs.  Additional arguments are forwarded to each
# script (for example, a list of input CSVs or --maxiter 2000).
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/flos-code-uv-cache}"
mkdir -p "$UV_CACHE_DIR"

uv run python run_knowledge_perception_profession_report.py "$@"

git add -A && \
	git commit -m "Update reports" && \
	git push

uv run python run_model_selection_random_slopes_report.py "$@"

git add -A && \
	git commit -m "Update reports" && \
	git push

uv run run_lasso_profession_selected_collocates_report.py "$@"

git add -A && \
	git commit -m "Update reports" && \
	git push
