#!/bin/bash --login
#SBATCH -p serial
#SBATCH -t 0-5

uv run count_professions.py --input ../corpora/dolma/dolma_3_sample.jsonl.gz \
	--max-documents 500000
