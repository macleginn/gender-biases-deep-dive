## Model comparison

After running `dissertation_analysing_results.py`, compare the fitted models across all
saved runs with:

```bash
uv run python compare_dissertation_model_results.py
```

This scans `analysis_outputs/` for run folders containing `run_summary.json` and writes:

- `analysis_outputs/model_comparisons/model_metrics_across_runs.csv`
- `analysis_outputs/model_comparisons/model_rankings.csv`
- `analysis_outputs/model_comparisons/model_summary.csv`

## Random-slope model selection report

Run backward random-slope model selection for every
`modelling_data/he_she_odds_results__*.csv`, aggregate the generated model data,
and build the HTML report with one command:

```bash
uv run python run_model_selection_random_slopes_report.py
```

To run a subset, pass CSVs explicitly:

```bash
uv run python run_model_selection_random_slopes_report.py he_she_odds_results__gpt2.csv
```

The starting fixed-effects model uses main effects only by default. Pass
`--starting-fixed-effect-interactions` if you want the starting model to include
pairwise fixed-effect interactions.

To rebuild the aggregate outputs and report from an existing `model_selection_data/`
run without refitting models:

```bash
uv run python run_model_selection_random_slopes_report.py --reuse-existing
```

The script writes all execution data used by the report to `model_selection_data/`:

- `inputs/`: copied input CSVs
- `runs/`: per-input prepared data, fit summaries, fixed effects, random effects, covariance matrices, selection traces, and run summaries
- `comparisons/`: aggregate metrics, best-model coefficients, best-model random effects, and selected-model summaries
- `execution_manifest.json`: inputs, options, run directories, and report path

The report is written to `model_selection_report/report.html`, with report figures
and HTML tables in sibling `figures/` and `tables/` directories.

## Profession-embedding model selection report

The companion script replaces profession random effects with profession embeddings.
It reads collocate counts from `profession_collocates.csv`, applies TF-IDF, computes
`k` truncated-SVD components (five by default), and joins those components to each
modelling CSV:

```bash
uv run python run_model_selection_profession_embeddings_report.py --k 5
```

The R-style starting formula is:

```r
log_he_she_odds ~ (tense + semantic_role + syntactic_role + valence +
  dominance + log(frequency) + lex_emb_norm)^2 +
  profession_embedding_1 + ... + profession_embedding_5
```

The embedding dimensions are ordinary fixed predictors. They are never included in
interactions, and the report treats all `k` dimensions together when decomposing
explained variance. Outputs are written to
`profession_embedding_model_selection_data/` and
`profession_embedding_model_selection_report/report.html`.
