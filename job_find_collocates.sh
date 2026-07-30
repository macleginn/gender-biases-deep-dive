#!/bin/bash --login
#SBATCH -p serial
#SBATCH -t 0-5

uv run count_profession_collocates.py \
	--input ../corpora/dolma/dolma_3_sample.jsonl.gz \
	--max-documents 500000 \
	--left-window-size 10 \
	--right-window-size 50
