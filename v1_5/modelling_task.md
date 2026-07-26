Modify @run_lasso_profession_embeddings_report.py to also build and report on the following models, in the same way it does now:

1. A "World-Knowledge Model". Predict he/she log odds based only on male-membership frequency (MalePerc from professions.csv),
semantic role, valence, and dominance, with interactions.
2. A "Human-Perception Model". Same as above but with syntactic role.

Take all professions from the first column in professions.csv.
