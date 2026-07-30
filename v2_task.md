Update run_model_selection_random_slopes_report.py to base model selection on the following full formula:


```
log_he_she_odds ~
  (tense + semantic_role + syntactic_role + valence + dominance + log(frequency) + lex_emb_norm)^2 +
  (1 + semantic_role + syntactic_role + valence + dominance | profession)
```
