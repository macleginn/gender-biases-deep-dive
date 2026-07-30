#!/usr/bin/env python3
"""Fit and report models that represent profession with SVD embeddings.

Only collocates listed in the gendered and names vocabulary files are used.
Four profession embeddings are computed from the resulting subset of
profession_collocates.csv and joined to each modelling data set: TF-IDF, raw
counts, log(1 + x) counts, and PPMI. Embedding dimensions are fixed effects,
not random effects. LassoCV selects predictors, which are then refit with OLS
and ranked by their unique contributions to R-squared.

R-style model formula (with k=10):
    log_he_she_odds ~ (tense + semantic_role + syntactic_role + valence +
                       dominance + log(frequency) + lex_emb_norm)^2 +
                       profession_embedding_1 + ... + profession_embedding_10
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import textwrap
import warnings
from datetime import date, datetime, timezone
from html import escape
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import statsmodels.formula.api as smf
    from patsy import dmatrices
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


REQUIRED_COLUMNS = {
    "tense",
    "syntactic_role",
    "semantic_role",
    "valence",
    "dominance",
    "frequency",
    "lex_emb_norm",
    "profession",
    "log_he_she_odds",
}
CATEGORICAL_COLUMNS = {
    "tense",
    "syntactic_role",
    "semantic_role",
    "valence",
    "dominance",
    "profession",
}
TREATMENT_REFERENCES = {"valence": "-val", "dominance": "-dom"}
FIXED_PREDICTORS = [
    "tense",
    "semantic_role",
    "syntactic_role",
    "valence",
    "dominance",
    "frequency",
    "lex_emb_norm",
]
NUMERICAL_PREDICTORS = ["log_frequency", "lex_emb_norm"]
PREPROCESSING_VERSION = (
    "log-frequency_zscore_and_frequency-residualized-lex-embedding_v1"
)
EMBEDDING_METHODS = {
    "tfidf": "TF-IDF",
    "raw_counts": "Raw counts",
    "log_counts": "log(1 + x) counts",
    "ppmi": "PPMI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path, nargs="*")
    parser.add_argument(
        "--profession-collocates", type=Path, default=Path("profession_collocates.csv")
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of SVD embedding dimensions (default: 10).",
    )
    parser.add_argument(
        "--collocate-list",
        type=Path,
        action="append",
        default=None,
        help="Text file of allowed collocates (may be supplied more than once).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("profession_selected_collocates_data")
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("profession_selected_collocates_report")
    )
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument(
        "--shapley-permutations",
        type=int,
        default=500,
        help="Number of random predictor orderings for Monte Carlo Shapley R² values (default: 500).",
    )
    parser.add_argument(
        "--shapley-random-state",
        type=int,
        default=0,
        help="Random seed for Monte Carlo Shapley R² values (default: 0).",
    )
    parser.add_argument("--starting-fixed-effect-interactions", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def clean_model_name(value: str) -> str:
    return re.sub(
        r"__[0-9a-f]{8}$", "", str(value).replace("he_she_odds_results__", "")
    )


def load_model_display_config() -> tuple[list[str], dict[str, str]]:
    model_ids = [
        line.strip().replace("/", "_")
        for line in Path("full_model_list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model_labels = [
        line.strip()
        for line in Path("model_names.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(model_ids) != len(model_labels):
        raise ValueError(
            "full_model_list.txt and model_names.txt must contain the same number of non-empty lines."
        )
    return model_ids, dict(zip(model_ids, model_labels))


def fixed_effect_label(term_name: str) -> str:
    if term_name == "Intercept":
        return "Intercept"
    label = re.sub(
        r"C\(([^,]+), Treatment\(reference='[^']+'\)\)", r"\1", str(term_name)
    )
    label = re.sub(r"C\(([^)]+)\)", r"\1", label)
    label = re.sub(r"\[T\.([^]]+)\]", r" (\1)", label)
    label = label.replace("_", " ")
    return label[:1].upper() + label[1:]


def term(name: str) -> str:
    if name == "frequency":
        return "log_frequency"
    if name in TREATMENT_REFERENCES:
        return f"C({name}, Treatment(reference='{TREATMENT_REFERENCES[name]}'))"
    return f"C({name})" if name in CATEGORICAL_COLUMNS else name


def fixed_terms(predictors: list[str], interactions: bool) -> list[str]:
    main = [term(name) for name in predictors]
    return main + (
        [f"{a}:{b}" for a, b in combinations(main, 2)] if interactions else []
    )


def standardize_predictors(
    df: pd.DataFrame, columns: list[str], *, source: Path
) -> dict[str, dict[str, float]]:
    """Z-score numeric predictors in place and return the fitted scaling values."""
    scaling: dict[str, dict[str, float]] = {}
    for column in columns:
        mean = float(df[column].mean())
        std = float(df[column].std(ddof=0))
        if not np.isfinite(std) or std == 0:
            raise ValueError(
                f"{source} has no usable variation in {column} for standardization."
            )
        df[column] = (df[column] - mean) / std
        scaling[column] = {"mean": mean, "std": std}
    return scaling


def load_results_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    for column in REQUIRED_COLUMNS:
        if column in CATEGORICAL_COLUMNS:
            df[column] = df[column].astype("string").str.strip().astype("category")
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if (df["frequency"] <= 0).any():
        raise ValueError(f"{path} contains non-positive frequency values")
    df["log_frequency"] = np.log(df["frequency"])
    df = df.dropna(subset=sorted(REQUIRED_COLUMNS)).copy()

    # Match the random-slopes preprocessing: lex_emb_norm represents lexical
    # embedding variation independent of log frequency, then both numeric
    # predictors are standardized for the fitted OLS models as well as Lasso.
    frequency_design = np.column_stack(
        [np.ones(len(df)), df["log_frequency"].to_numpy()]
    )
    lex_embedding = df["lex_emb_norm"].to_numpy()
    frequency_coefficients, *_ = np.linalg.lstsq(
        frequency_design, lex_embedding, rcond=None
    )
    df["lex_emb_norm"] = lex_embedding - frequency_design @ frequency_coefficients
    scaling = standardize_predictors(df, NUMERICAL_PREDICTORS, source=path)
    df.attrs["preprocessing"] = {
        "version": PREPROCESSING_VERSION,
        "lex_emb_norm": {
            "operation": "OLS residual from lex_emb_norm ~ 1 + log_frequency",
            "intercept": float(frequency_coefficients[0]),
            "log_frequency_coefficient": float(frequency_coefficients[1]),
        },
        "standardized_predictors": scaling,
    }
    return df


def load_allowed_collocates(paths: list[Path]) -> set[str]:
    """Load and combine newline-delimited collocate vocabularies."""
    collocates: set[str] = set()
    for path in paths:
        collocates.update(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not collocates:
        raise ValueError("The collocate vocabulary files contain no non-empty entries")
    return collocates


def compute_profession_embeddings(
    path: Path,
    k: int,
    method: str = "tfidf",
    allowed_collocates: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transform collocate counts and return k SVD coordinates per profession.

    PPMI is computed over the profession-by-collocate count matrix using the
    usual marginal-distribution PMI definition, with negative PMI values set
    to zero.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if method not in EMBEDDING_METHODS:
        raise ValueError(
            f"Unknown embedding method {method!r}; choose from {', '.join(EMBEDDING_METHODS)}"
        )
    collocates = pd.read_csv(path)
    if collocates.shape[1] < 2:
        raise ValueError(
            f"{path} must contain a profession column and collocate columns"
        )
    profession_column = collocates.columns[0]
    professions = collocates[profession_column].astype("string").str.strip()
    if professions.isna().any() or professions.duplicated().any():
        raise ValueError(f"{path} contains missing or duplicate profession names")
    collocate_columns = list(collocates.columns[1:])
    if allowed_collocates is not None:
        selected_columns = [
            column for column in collocate_columns if column in allowed_collocates
        ]
        if not selected_columns:
            raise ValueError(
                f"None of the collocates in {path} are present in the supplied vocabularies"
            )
        collocates = collocates[[profession_column, *selected_columns]]
        collocate_columns = selected_columns
    counts = (
        collocates.loc[:, collocate_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    if (counts < 0).any().any():
        raise ValueError(f"{path} contains negative collocate counts")
    if len(collocates) < 2:
        raise ValueError("At least two professions are required for SVD")
    count_matrix = counts.to_numpy(dtype=float)
    if not np.isfinite(count_matrix).all():
        raise ValueError(f"{path} contains non-finite collocate counts")
    if count_matrix.sum() <= 0:
        raise ValueError(f"{path} contains no positive collocate counts")
    if method == "tfidf":
        transformed = TfidfTransformer().fit_transform(count_matrix)
    elif method == "raw_counts":
        transformed = count_matrix
    elif method == "log_counts":
        transformed = np.log1p(count_matrix)
    else:
        total = count_matrix.sum()
        row_totals = count_matrix.sum(axis=1, keepdims=True)
        column_totals = count_matrix.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((count_matrix * total) / (row_totals * column_totals))
        pmi = np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0)
        transformed = np.maximum(pmi, 0.0)
        transformed[count_matrix == 0] = 0.0
    max_components = min(transformed.shape[0] - 1, transformed.shape[1])
    if k > max_components:
        raise ValueError(
            f"k={k} exceeds the maximum available SVD components ({max_components})"
        )
    coordinates = TruncatedSVD(n_components=k, random_state=0).fit_transform(
        transformed
    )
    columns = [f"profession_embedding_{index}" for index in range(1, k + 1)]
    embedding = pd.DataFrame(coordinates, columns=columns)
    embedding.insert(0, "profession", professions.to_numpy())
    return embedding, {
        "source": str(path.resolve()),
        "profession_column": str(profession_column),
        "rows": int(len(embedding)),
        "collocate_columns": int(counts.shape[1]),
        "selected_collocates": collocate_columns,
        "k": k,
        "columns": columns,
        "method": method,
        "method_label": EMBEDDING_METHODS[method],
    }


def add_embedding(df: pd.DataFrame, embedding: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["profession"] = result["profession"].astype("string").str.strip()
    merged = result.merge(
        embedding, on="profession", how="left", validate="many_to_one"
    )
    missing = int(merged[embedding.columns[1]].isna().sum())
    if missing:
        unknown = sorted(
            result.loc[merged[embedding.columns[1]].isna(), "profession"].unique()
        )
        raise ValueError(
            f"{missing} rows have professions absent from the embedding: {unknown[:10]}"
        )
    return merged


def fixed_variances(result: Any, embedding_columns: list[str]) -> dict[str, float]:
    """Allocate fitted fixed-effect variance, including between-term covariance.

    Each term receives the variance of its fitted contribution plus its
    covariance with every other term.  This divides each pairwise covariance
    equally between the two terms, so the reported allocations sum to the
    variance of the complete fixed-effects linear predictor (apart from
    floating-point rounding).  Profession-embedding dimensions are treated as
    one grouped term before making that allocation.
    """
    info = result.model.data.design_info
    design = np.asarray(result.model.exog)
    beta = np.asarray(result.params)
    term_names: list[str] = []
    contributions: list[np.ndarray] = []
    embedding_slices = [
        info.term_name_slices[column]
        for column in embedding_columns
        if column in info.term_name_slices
    ]
    if embedding_slices:
        embedding_design = np.concatenate(
            [design[:, sl] for sl in embedding_slices], axis=1
        )
        embedding_beta = np.concatenate([beta[sl] for sl in embedding_slices])
        term_names.append("profession_embedding")
        contributions.append(embedding_design @ embedding_beta)
    for name, sl in info.term_name_slices.items():
        if name == "Intercept" or name in embedding_columns:
            continue
        term_names.append(name)
        contributions.append(design[:, sl] @ beta[sl])
    if not contributions:
        return {}
    contribution_matrix = np.column_stack(contributions)
    covariance = np.atleast_2d(np.cov(contribution_matrix, rowvar=False, ddof=1))
    allocations = covariance.sum(axis=1)
    return {
        f"fixed_effect_variance_{name}": float(allocation)
        for name, allocation in zip(term_names, allocations)
    }


def fit_model(
    df: pd.DataFrame,
    name: str,
    terms: list[str],
    embedding_columns: list[str],
    run_dir: Path,
) -> dict[str, Any]:
    model_dir = run_dir / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    formula = "log_he_she_odds ~ " + (" + ".join(terms) if terms else "1")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = smf.ols(formula=formula, data=df).fit()
        fixed = pd.DataFrame(
            {
                "term": result.params.index,
                "coef": result.params,
                "std_err": result.bse,
                "t": result.tvalues,
                "p_value": result.pvalues,
                "ci_low": result.conf_int()[0],
                "ci_high": result.conf_int()[1],
            }
        )
        fixed.to_csv(model_dir / "fixed_effects.csv", index=False)
        metrics = {
            "formula": formula,
            "model_type": "OLS",
            "optimizer": "closed_form",
            "converged": True,
            "nobs": int(result.nobs),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "log_likelihood": float(result.llf),
            "residual_variance": float(result.scale),
            "R2m": float(result.rsquared),
            "R2c": float(result.rsquared),
            "warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
            **fixed_variances(result, embedding_columns),
        }
        (model_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        (model_dir / "summary.txt").write_text(
            result.summary().as_text(), encoding="utf-8"
        )
        return {
            "status": "ok",
            "result": result,
            "metrics": metrics,
            "terms": terms,
            "name": name,
        }
    except Exception as exc:
        (model_dir / "fit_failed.json").write_text(
            json.dumps({"error": repr(exc)}, indent=2), encoding="utf-8"
        )
        return {"status": "failed", "error": repr(exc), "name": name, "terms": terms}


def decompose_r_squared(
    df: pd.DataFrame, selected: dict[str, Any], terms: list[str], run_dir: Path
) -> pd.DataFrame:
    """Estimate each selected term's unique R² using drop-one refits."""
    full_result = selected["result"]
    rows = []
    for predictor in terms:
        reduced_terms = [term_name for term_name in terms if term_name != predictor]
        formula = "log_he_she_odds ~ " + (
            " + ".join(reduced_terms) if reduced_terms else "1"
        )
        reduced_result = smf.ols(formula=formula, data=df).fit()
        unique_r2 = max(0.0, float(full_result.rsquared - reduced_result.rsquared))
        rows.append({"predictor": predictor, "unique_r2": unique_r2})
    decomposition = pd.DataFrame(rows)
    total_unique_r2 = decomposition["unique_r2"].sum()
    decomposition["relative_r2"] = (
        decomposition["unique_r2"] / total_unique_r2 if total_unique_r2 > 0 else 0.0
    )
    decomposition = decomposition.sort_values("unique_r2", ascending=False).reset_index(
        drop=True
    )
    decomposition.to_csv(run_dir / "r2_decomposition.csv", index=False)
    return decomposition


def decompose_shapley_r_squared(
    df: pd.DataFrame,
    selected: dict[str, Any],
    terms: list[str],
    embedding_columns: list[str],
    run_dir: Path,
    permutations: int,
    random_state: int,
) -> pd.DataFrame:
    """Approximate grouped Shapley values using model R² as the payout.

    A sampled ordering adds predictors one at a time.  Its marginal increase
    in R² is that predictor's payout for the ordering; averaging over random
    orderings estimates its Shapley value.  The SVD coordinates are evaluated
    together as one ``profession_embedding`` predictor, matching the
    covariance-adjusted variance decomposition.
    """
    if permutations < 1:
        raise ValueError("shapley permutations must be at least 1")

    embedding_terms = [term_name for term_name in terms if term_name in embedding_columns]
    predictor_groups = [
        ("profession_embedding", embedding_terms)
    ] if embedding_terms else []
    predictor_groups.extend(
        (term_name, [term_name])
        for term_name in terms
        if term_name not in embedding_columns
    )
    predictor_names = [name for name, _ in predictor_groups]
    if not predictor_names:
        return pd.DataFrame(
            columns=[
                "predictor",
                "shapley_r2",
                "mc_standard_error",
                "relative_r2",
                "permutations",
                "random_state",
                "baseline_r2",
                "full_r2",
            ]
        )

    utility_cache: dict[tuple[int, ...], float] = {(): 0.0}

    def utility(included_indices: tuple[int, ...]) -> float:
        if included_indices not in utility_cache:
            included_terms = [
                formula_term
                for index in included_indices
                for formula_term in predictor_groups[index][1]
            ]
            formula = "log_he_she_odds ~ " + " + ".join(included_terms)
            utility_cache[included_indices] = float(smf.ols(formula=formula, data=df).fit().rsquared)
        return utility_cache[included_indices]

    rng = np.random.default_rng(random_state)
    marginal_r2 = np.empty((permutations, len(predictor_groups)))
    for sample in range(permutations):
        ordering = rng.permutation(len(predictor_groups))
        included: tuple[int, ...] = ()
        previous_utility = utility(included)
        for index in ordering:
            included = tuple(sorted((*included, int(index))))
            current_utility = utility(included)
            marginal_r2[sample, index] = current_utility - previous_utility
            previous_utility = current_utility

    shapley_r2 = marginal_r2.mean(axis=0)
    standard_error = (
        marginal_r2.std(axis=0, ddof=1) / np.sqrt(permutations)
        if permutations > 1
        else np.zeros(len(predictor_groups))
    )
    baseline_r2 = utility(())
    full_r2 = float(selected["result"].rsquared)
    total_payout = full_r2 - baseline_r2
    decomposition = pd.DataFrame(
        {
            "predictor": predictor_names,
            "shapley_r2": shapley_r2,
            "mc_standard_error": standard_error,
            "relative_r2": shapley_r2 / total_payout if total_payout > 0 else 0.0,
            "permutations": permutations,
            "random_state": random_state,
            "baseline_r2": baseline_r2,
            "full_r2": full_r2,
        }
    ).sort_values("shapley_r2", ascending=False, ignore_index=True)
    decomposition.to_csv(run_dir / "shapley_r2_decomposition.csv", index=False)
    return decomposition


def fold_shapley_interactions(decomposition: pd.DataFrame) -> pd.DataFrame:
    """Allocate each interaction's Shapley R² equally to its component terms."""
    columns = ["target_model", "predictor", "folded_shapley_r2"]
    if decomposition.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in decomposition.itertuples(index=False):
        components = str(row.predictor).split(":")
        for component in components:
            rows.append(
                {
                    "target_model": row.target_model,
                    "predictor": component,
                    "folded_shapley_r2": row.shapley_r2 / len(components),
                }
            )
    return pd.DataFrame(rows).groupby(
        ["target_model", "predictor"], as_index=False
    )["folded_shapley_r2"].sum()


def select_model(
    df: pd.DataFrame,
    run_dir: Path,
    embedding_columns: list[str],
    maxiter: int,
    interactions: bool,
    shapley_permutations: int,
    shapley_random_state: int,
) -> tuple[dict[str, Any], list[str], pd.DataFrame]:
    all_terms = fixed_terms(FIXED_PREDICTORS, interactions) + embedding_columns
    formula = "log_he_she_odds ~ " + " + ".join(all_terms)
    response, design = dmatrices(formula, data=df, return_type="dataframe")
    feature_names = list(design.columns)
    x_scaled = StandardScaler().fit_transform(design)
    y = np.asarray(response).ravel()
    lasso = LassoCV(cv=5, max_iter=maxiter, n_jobs=-1, random_state=0).fit(x_scaled, y)

    pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": lasso.coef_,
            "nonzero": np.abs(lasso.coef_) > 1e-10,
        }
    ).to_csv(run_dir / "lasso_coefficients.csv", index=False)

    # Categorical and interaction terms expand into multiple design columns.
    # Keep a formula term when any of its expanded columns is non-zero.
    selected_terms = []
    for term_name, term_slice in design.design_info.term_name_slices.items():
        if term_name == "Intercept":
            continue
        nonzero = np.abs(lasso.coef_[term_slice]) > 1e-10
        if nonzero.any():
            selected_terms.append(term_name)

    selected = fit_model(
        df, "selected_profession_embedding", selected_terms, embedding_columns, run_dir
    )

    r2_decomposition = decompose_r_squared(df, selected, selected_terms, run_dir)
    decompose_shapley_r_squared(
        df,
        selected,
        selected_terms,
        embedding_columns,
        run_dir,
        shapley_permutations,
        shapley_random_state,
    )

    (run_dir / "selected_model.json").write_text(
        json.dumps(
            {
                "selected_model": selected["name"],
                "selected_terms": selected_terms,
                "alpha": float(lasso.alpha_),
                "alpha_selection": "LassoCV",
                "cv": 5,
                "lasso_n_iter": int(lasso.n_iter_),
                "shapley_permutations": shapley_permutations,
                "shapley_random_state": shapley_random_state,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return selected, selected_terms, r2_decomposition


def run_one(
    input_csv: Path,
    embedding: pd.DataFrame,
    embedding_meta: dict[str, Any],
    data_dir: Path,
    maxiter: int,
    interactions: bool,
    method: str,
    shapley_permutations: int,
    shapley_random_state: int,
) -> Path:
    run_dir = data_dir / "runs" / method / slugify(input_csv.stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_csv, data_dir / "inputs" / input_csv.name)
    results = load_results_csv(input_csv)
    preprocessing = results.attrs.get("preprocessing", {})
    df = add_embedding(results, embedding)
    df.to_csv(run_dir / "prepared_results.csv", index=False)
    embedding_columns = list(embedding.columns[1:])
    selected, terms, r2_decomposition = select_model(
        df,
        run_dir,
        embedding_columns,
        maxiter,
        interactions,
        shapley_permutations,
        shapley_random_state,
    )
    baseline = fit_model(
        df,
        "no_profession_embedding_baseline",
        fixed_terms(FIXED_PREDICTORS, interactions),
        embedding_columns,
        run_dir,
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "input_csv": str(input_csv.resolve()),
                "embedding": embedding_meta,
                "selected_terms": terms,
                "selected_predictor_count": int(len(r2_decomposition)),
                "selected_model": selected["name"],
                "baseline_model": baseline["name"],
                "preprocessing": preprocessing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def discover_inputs(paths: list[Path]) -> list[Path]:
    inputs = paths or sorted(Path("modelling_data").glob("he_she_odds_results__*.csv"))
    if not inputs:
        raise FileNotFoundError("No input CSVs found")
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input CSVs: " + ", ".join(missing))
    return [p.resolve() for p in inputs]


def write_table(df: pd.DataFrame, path: Path, *, classes: str = "data-table") -> str:
    html = df.to_html(
        index=False,
        border=0,
        classes=classes,
        escape=True,
        float_format=lambda value: f"{value:.3f}",
    )
    collapsed_html = (
        '<details class="table-details"><summary>Show table</summary>'
        f'<div class="table-wrap">{html}</div></details>'
    )
    path.write_text(collapsed_html, encoding="utf-8")
    return collapsed_html


def write_grouped_table(
    df: pd.DataFrame,
    path: Path,
    groups: pd.Series,
    group_order: list[str],
    group_labels: dict[str, str],
    *,
    classes: str = "data-table",
) -> str:
    """Write one vertically stacked HTML table for each preprocessing method."""
    sections = []
    for group in group_order:
        subset = df.loc[groups.to_numpy() == group]
        if subset.empty:
            continue
        table = subset.to_html(
            index=False,
            border=0,
            classes=classes,
            escape=True,
            float_format=lambda value: f"{value:.3f}",
        )
        sections.append(
            f'<details class="table-details"><summary>{escape(group_labels[group])} table'
            f'</summary><div class="table-wrap">{table}</div></details>'
        )
    html = "\n".join(sections)
    path.write_text(html, encoding="utf-8")
    return html


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def figure(src: str, caption: str) -> str:
    return (
        f'<figure><img src="{escape(src)}" alt="{escape(caption)}" loading="lazy">'
        f"<figcaption>{escape(caption)}</figcaption></figure>"
    )


def build_report(
    run_dirs: list[Path], report_dir: Path, embedding_meta: dict[str, Any]
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir / "figures"
    table_dir = report_dir / "tables"
    figure_dir.mkdir(exist_ok=True)
    table_dir.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    for run in run_dirs:
        summary = json.loads((run / "run_summary.json").read_text())
        for model_dir in sorted((run / "models").iterdir()):
            metrics_path = model_dir / "metrics.json"
            if metrics_path.exists():
                rows.append(
                    {
                        "target_model": clean_model_name(
                            Path(summary["input_csv"]).stem
                        ),
                        "embedding_method": summary["embedding"].get("method", "tfidf"),
                        "embedding_method_label": summary["embedding"].get(
                            "method_label", "TF-IDF"
                        ),
                        "model_name": model_dir.name,
                        "model_dir": str(model_dir),
                        **json.loads(metrics_path.read_text()),
                    }
                )
    all_metrics = pd.DataFrame(rows)
    all_metrics.to_csv(report_dir / "all_model_metrics.csv", index=False)
    selected = all_metrics.loc[
        all_metrics["model_name"] == "selected_profession_embedding"
    ].copy()
    selected.to_csv(report_dir / "model_metrics.csv", index=False)

    if selected.empty:
        raise ValueError(
            "No selected profession-embedding models were found in the supplied runs"
        )

    selected["target_model"] = selected["target_model"].map(clean_model_name)
    selected["model_label"] = (
        selected["target_model"] + " — " + selected["embedding_method_label"]
    )
    selected = selected.sort_values(["target_model", "embedding_method"]).reset_index(
        drop=True
    )
    fixed_rows = []
    for _, row in selected.iterrows():
        fixed = pd.read_csv(Path(row["model_dir"]) / "fixed_effects.csv")
        fixed.insert(0, "target_model", row["model_label"])
        fixed_rows.append(fixed)
    coefficients = (
        pd.concat(fixed_rows, ignore_index=True) if fixed_rows else pd.DataFrame()
    )
    coefficients.to_csv(report_dir / "selected_coefficients.csv", index=False)
    r2_rows = []
    for _, row in selected.iterrows():
        decomposition_path = (
            Path(row["model_dir"]).parent.parent / "r2_decomposition.csv"
        )
        if decomposition_path.exists():
            decomposition = pd.read_csv(decomposition_path)
            decomposition.insert(0, "target_model", row["model_label"])
            r2_rows.append(decomposition)
    r2_decomposition = (
        pd.concat(r2_rows, ignore_index=True) if r2_rows else pd.DataFrame()
    )
    r2_decomposition.to_csv(report_dir / "r2_decomposition.csv", index=False)
    shapley_rows = []
    for _, row in selected.iterrows():
        decomposition_path = (
            Path(row["model_dir"]).parent.parent / "shapley_r2_decomposition.csv"
        )
        if decomposition_path.exists():
            decomposition = pd.read_csv(decomposition_path)
            decomposition.insert(0, "target_model", row["model_label"])
            shapley_rows.append(decomposition)
    shapley_r2_decomposition = (
        pd.concat(shapley_rows, ignore_index=True) if shapley_rows else pd.DataFrame()
    )
    shapley_r2_decomposition.to_csv(
        report_dir / "shapley_r2_decomposition.csv", index=False
    )
    folded_shapley = fold_shapley_interactions(shapley_r2_decomposition)
    folded_shapley.to_csv(report_dir / "shapley_r2_folded.csv", index=False)
    variance_cols = [
        c for c in selected.columns if c.startswith("fixed_effect_variance_")
    ]
    variance = selected[["model_label", *variance_cols]].copy()
    variance = variance.rename(
        columns={
            "model_label": "Model",
            "fixed_effect_variance_profession_embedding": "Profession embedding",
            **{
                column: column.removeprefix("fixed_effect_variance_").replace("_", " ")
                for column in variance_cols
                if column != "fixed_effect_variance_profession_embedding"
            },
        }
    )
    available_models = set(selected["model_label"])
    configured_order, configured_labels = load_model_display_config()
    target_order = [
        model for model in configured_order if model in set(selected["target_model"])
    ]
    target_order.extend(sorted(set(selected["target_model"]) - set(target_order)))
    model_order = [
        f"{target} — {label}"
        for target in target_order
        for label in EMBEDDING_METHODS.values()
        if f"{target} — {label}" in available_models
    ]
    model_order.extend(sorted(available_models - set(model_order)))
    plot_model_labels = {}
    plot_model_methods = {}
    for model in model_order:
        target, method_label = model.split(" — ", 1)
        plot_model_labels[model] = (
            f"{configured_labels.get(target, target)}\n{method_label}"
        )
        plot_model_methods[plot_model_labels[model]] = next(
            method
            for method, label in EMBEDDING_METHODS.items()
            if label == method_label
        )
    plot_model_order = [plot_model_labels[model] for model in model_order]
    plot_model_order_reversed = plot_model_order[::-1]
    method_order = [
        method
        for method in EMBEDDING_METHODS
        if method in set(selected["embedding_method"])
    ]
    method_labels = {method: EMBEDDING_METHODS[method] for method in method_order}
    variance["Model"] = variance["Model"].map(plot_model_labels)
    variance = variance.set_index("Model").reindex(plot_model_order).reset_index()
    variance.columns = [
        "Model" if column == "Model" else fixed_effect_label(column)
        for column in variance.columns
    ]
    coefficients["term"] = coefficients["term"].map(fixed_effect_label)
    coefficients["target_model"] = coefficients["target_model"].map(plot_model_labels)
    variance.to_csv(report_dir / "variance_decomposition.csv", index=False)

    fit_table = selected[
        [
            "model_label",
            "converged",
            "nobs",
            "aic",
            "bic",
            "log_likelihood",
            "R2m",
            "residual_variance",
        ]
    ].rename(
        columns={
            "model_label": "Model",
            "converged": "Converged",
            "nobs": "N",
            "aic": "AIC",
            "bic": "BIC",
            "log_likelihood": "Log likelihood",
            "R2m": "R2",
            "residual_variance": "Residual variance",
        }
    )
    fit_table["Model"] = fit_table["Model"].map(plot_model_labels)
    fit_groups = fit_table["Model"].map(plot_model_methods)
    variance_groups = variance["Model"].map(plot_model_methods)
    fit_html = write_grouped_table(
        fit_table,
        table_dir / "selected_model_fit.html",
        fit_groups,
        method_order,
        method_labels,
    )
    variance_html = write_grouped_table(
        variance,
        table_dir / "variance_decomposition.html",
        variance_groups,
        method_order,
        method_labels,
    )
    coefficient_table = coefficients.rename(
        columns={
            "target_model": "Model",
            "term": "Term",
            "coef": "Estimate",
            "std_err": "Std. error",
            "p_value": "p value",
            "ci_low": "CI low",
            "ci_high": "CI high",
        }
    )
    coefficient_groups = coefficient_table["Model"].map(plot_model_methods)
    coefficient_html = write_grouped_table(
        coefficient_table,
        table_dir / "selected_coefficients.html",
        coefficient_groups,
        method_order,
        method_labels,
        classes="data-table compact",
    )
    r2_table = r2_decomposition.copy()
    if not r2_table.empty:
        r2_table["target_model"] = r2_table["target_model"].map(plot_model_labels)
        r2_table["predictor"] = r2_table["predictor"].map(fixed_effect_label)
    r2_html = (
        write_grouped_table(
            r2_table,
            table_dir / "r2_decomposition.html",
            r2_table["target_model"].map(plot_model_methods),
            method_order,
            method_labels,
            classes="data-table compact",
        )
        if not r2_decomposition.empty
        else "<p>No R² decomposition results.</p>"
    )
    shapley_table = shapley_r2_decomposition.copy()
    if not shapley_table.empty:
        shapley_table["target_model"] = shapley_table["target_model"].map(
            plot_model_labels
        )
        shapley_table["predictor"] = shapley_table["predictor"].map(
            lambda name: (
                "Profession embedding"
                if name == "profession_embedding"
                else fixed_effect_label(name)
            )
        )
    shapley_html = (
        write_grouped_table(
            shapley_table,
            table_dir / "shapley_r2_decomposition.html",
            shapley_table["target_model"].map(plot_model_methods),
            method_order,
            method_labels,
            classes="data-table compact",
        )
        if not shapley_table.empty
        else "<p>No Monte Carlo Shapley R² results.</p>"
    )

    sns.set_theme(style="whitegrid", context="notebook")
    fit_table = fit_table.set_index("Model").reindex(plot_model_order).reset_index()
    target_plot_order = [
        configured_labels.get(target, target) for target in target_order
    ]
    fit_table["Target model"] = fit_table["Model"].str.split("\n").str[0]
    fit_table["Preprocessing"] = fit_table["Model"].map(plot_model_methods)
    fig, axes = plt.subplots(
        len(method_order),
        1,
        sharex=True,
        sharey=True,
        figsize=(max(10, len(target_plot_order) * 0.8), 2.8 * len(method_order)),
        squeeze=False,
    )
    for axis, method in zip(axes[:, 0], method_order):
        method_fit = fit_table.loc[fit_table["Preprocessing"] == method]
        sns.barplot(
            data=method_fit,
            x="Target model",
            y="R2",
            order=target_plot_order,
            color="#315c70",
            width=0.9,
            ax=axis,
        )
        axis.set_title(method_labels[method], loc="left", fontweight="bold")
        axis.set_ylim(0, 1.0)
        axis.set_xlabel("")
        axis.set_ylabel("Explained variance (R²)")
        axis.tick_params(axis="x", labelrotation=45)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
    axes[-1, 0].set_xlabel("Target language model")
    savefig(figure_dir / "r2_comparison.png")

    variance_plot = variance.set_index("Model")
    variance_plot = variance_plot.div(
        variance_plot.sum(axis=1).replace(0, np.nan), axis=0
    )
    fixed_effect_names = variance_plot.columns.tolist()
    fixed_effect_numbers = list(range(1, len(fixed_effect_names) + 1))
    fixed_effect_colors = sns.color_palette("husl", n_colors=len(fixed_effect_names))
    fig, axes = plt.subplots(
        len(method_order),
        1,
        sharex=True,
        figsize=(13, max(5.5, len(model_order) * 0.9 + 2.5)),
        squeeze=False,
    )
    for axis, method in zip(axes[:, 0], method_order):
        method_models = [
            label for label in plot_model_order if plot_model_methods[label] == method
        ]
        variance_plot.loc[method_models[::-1]].plot(
            kind="barh",
            stacked=True,
            ax=axis,
            color=fixed_effect_colors,
            width=0.9,
        )
        axis.set_title(method_labels[method], loc="left", fontweight="bold")
        axis.set_xlabel("Proportion of covariance-adjusted fixed-effect variance")
        axis.set_ylabel("")
        for effect_number, container in zip(fixed_effect_numbers, axis.containers):
            axis.bar_label(
                container,
                labels=[
                    (
                        str(effect_number)
                        if np.isfinite(bar.get_width()) and bar.get_width() >= 0.025
                        else ""
                    )
                    for bar in container
                ],
                label_type="center",
                color="#111111",
                fontsize=8,
                fontweight="bold",
            )
    handles, _ = axes[0, 0].get_legend_handles_labels()
    axes[-1, 0].legend(
        handles,
        [
            f"{number}. {name}"
            for number, name in zip(fixed_effect_numbers, fixed_effect_names)
        ],
        title="Fixed effect",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        ncols=min(3, len(fixed_effect_names)),
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.22)
    savefig(figure_dir / "variance_decomposition.png")

    shapley_figure_html = ""
    if not folded_shapley.empty:
        shapley_plot = folded_shapley.copy()
        shapley_plot["Model"] = shapley_plot["target_model"].map(plot_model_labels)
        shapley_plot["Predictor"] = shapley_plot["predictor"].map(
            lambda name: (
                "Profession embedding"
                if name == "profession_embedding"
                else fixed_effect_label(name)
            )
        )
        shapley_plot = shapley_plot.pivot_table(
            index="Model",
            columns="Predictor",
            values="folded_shapley_r2",
            aggfunc="sum",
            fill_value=0.0,
        ).reindex(plot_model_order, fill_value=0.0)
        shapley_predictors = shapley_plot.sum(axis=0).sort_values(ascending=False).index
        shapley_plot = shapley_plot.reindex(columns=shapley_predictors)
        shapley_colors = sns.color_palette("husl", n_colors=len(shapley_predictors))
        fig, axes = plt.subplots(
            len(method_order),
            1,
            sharex=True,
            figsize=(13, max(5.5, len(model_order) * 0.9 + 2.5)),
            squeeze=False,
        )
        for axis, method in zip(axes[:, 0], method_order):
            method_models = [
                label for label in plot_model_order if plot_model_methods[label] == method
            ]
            shapley_plot.loc[method_models[::-1]].plot(
                kind="barh", stacked=True, ax=axis, color=shapley_colors, width=0.9
            )
            axis.set_title(method_labels[method], loc="left", fontweight="bold")
            axis.set_xlabel("Shapley R² contribution (interactions split equally)")
            axis.set_ylabel("")
        handles, _ = axes[0, 0].get_legend_handles_labels()
        axes[-1, 0].legend(
            handles,
            shapley_predictors,
            title="Predictor",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            ncols=min(3, len(shapley_predictors)),
            fontsize=8,
        )
        fig.subplots_adjust(bottom=0.22)
        savefig(figure_dir / "shapley_r2_folded.png")
        shapley_figure_html = figure(
            "figures/shapley_r2_folded.png",
            "Shapley R² contributions by selected model, with each interaction split equally between its component predictors.",
        )

    fig, axes = plt.subplots(
        len(method_order),
        1,
        sharex=False,
        sharey=True,
        figsize=(10, max(4, 0.35 * coefficients["term"].nunique() * len(method_order))),
        squeeze=False,
    )
    for axis, method in zip(axes[:, 0], method_order):
        method_models = [
            label for label in plot_model_order if plot_model_methods[label] == method
        ]
        method_coefficients = coefficients.loc[
            coefficients["target_model"].isin(method_models)
        ]
        coefficient_plot = method_coefficients.pivot(
            index="term", columns="target_model", values="coef"
        ).reindex(columns=method_models)
        sns.heatmap(
            coefficient_plot,
            center=0,
            cmap="RdBu_r",
            ax=axis,
            cbar_kws={"label": "Coefficient"},
        )
        axis.set_title(method_labels[method], loc="left", fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("Fixed-effect term")
    savefig(figure_dir / "coefficient_heatmap.png")

    cards = [
        ("Target models", str(len(selected))),
        ("Embedding dimensions", str(embedding_meta["k"])),
        ("Observations/model", f"{int(selected['nobs'].median()):,}"),
        (
            "Selected models converged",
            "yes" if bool(selected["converged"].all()) else "no",
        ),
    ]
    cards_html = "\n".join(
        f'<div class="metric-card"><div class="metric-value">{escape(value)}</div>'
        f'<div class="metric-label">{escape(label)}</div></div>'
        for label, value in cards
    )
    html = textwrap.dedent(f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Profession-Embedding Model Selection Report</title>
          <style>
            :root {{ --bg:#f4efe6; --panel:#fffaf1; --ink:#1d2625; --muted:#64706b; --border:#d9c8ad; --accent:#315c70; }}
            body {{ margin:0; color:var(--ink); font-family:Charter, Georgia, serif; background:radial-gradient(circle at 8% 0%,#ecd0bc,transparent 28rem),radial-gradient(circle at 95% 10%,#c9dce2,transparent 26rem),var(--bg); }}
            .page {{ max-width:1220px; margin:0 auto; padding:32px 20px 60px; }}
            header,section {{ background:rgba(255,250,241,.93); border:1px solid var(--border); border-radius:24px; padding:26px; margin-bottom:22px; }}
            h1 {{ font-size:clamp(2.2rem,5vw,4.5rem); line-height:.96; margin:0 0 12px; }} h2 {{ margin:0 0 12px; font-size:1.7rem; }}
            .subtitle,.section-note,figcaption {{ color:var(--muted); line-height:1.5; }}
            .metric-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-top:20px; }}
            .metric-card {{ border:1px solid var(--border); border-radius:16px; padding:14px; background:#fffdf8; }} .metric-value {{ font-size:1.7rem; font-weight:700; color:var(--accent); }}
            .figure-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:16px; }} figure {{ margin:16px auto; }} img {{ width:100%; border:1px solid var(--border); border-radius:14px; background:white; }}
            .table-details {{ border:1px solid var(--border); border-radius:14px; margin:12px 0 18px; background:#fffdf8; }} .table-details summary {{ cursor:pointer; padding:10px 14px; color:var(--accent); font-weight:700; }} .table-details[open] summary {{ border-bottom:1px solid var(--border); }} .table-wrap {{ overflow-x:auto; background:white; }} .data-table {{ width:100%; border-collapse:collapse; font-size:.92rem; }}
            .data-table th,.data-table td {{ border:1px solid #e2d4bf; padding:8px 10px; text-align:left; vertical-align:top; white-space:nowrap; }} .data-table th {{ background:#f0dfc7; }} .compact {{ font-size:.84rem; }}
            @media (max-width:760px) {{ .page {{ padding:14px 10px 36px; }} header,section {{ padding:16px; }} .figure-grid {{ grid-template-columns:1fr; }} }}
          </style>
        </head>
        <body><div class="page">
          <header><h1>Profession-Embedding Model Selection Report</h1>
            <p class="subtitle">LassoCV-selected and OLS-refit models for log he/she odds. Each profession representation uses {escape(str(embedding_meta['k']))} SVD dimensions, comparing TF-IDF, raw counts, log(1 + x) counts, and PPMI. <code>lex_emb_norm</code> is residualized on <code>log_frequency</code>, and both are z-scored before fitting. Generated on {date.today().isoformat()}.</p>
            <div class="metric-grid">{cards_html}</div>
          </header>
          <section><h2>Selected Model Fit</h2><p class="section-note">One final selected configuration is reported for each target model. All candidate fits remain available in <code>all_model_metrics.csv</code>.</p>{fit_html}{figure("figures/r2_comparison.png", "Explained variance (R2) by selected model.")}</section>
          <section><h2>Fixed-Effect Variance</h2><p class="section-note">The embedding dimensions are grouped as one predictor; they are not included in interactions. Each term's allocation includes its fitted-contribution variance plus half of every pairwise covariance, so allocations sum to the variance of the complete fixed-effects predictor. Negative allocations can occur when a term covaries negatively with the others. Segment numbers in the horizontal bars map to the indexed legend below the plot.</p>{variance_html}{figure("figures/variance_decomposition.png", "Proportion of covariance-adjusted fixed-effect variance by selected model; segment numbers map to the indexed legend.")}</section>
          <section><h2>Selected Coefficients</h2>{figure("figures/coefficient_heatmap.png", "Fixed-effect coefficients by selected model.")}{coefficient_html}</section>
          <section><h2>R² Decomposition</h2><p class="section-note">Each value is the unique R² attributable to a selected predictor, estimated by the drop in R² after refitting the OLS model without that predictor. Relative R² is normalized across the selected predictors. The Lasso penalty alpha is selected by five-fold cross-validation.</p>{r2_html}</section>
          <section><h2>Monte Carlo Shapley R² Attribution</h2><p class="section-note">R² is the payout. For each random predictor ordering, a predictor receives the increase in R² when it enters the OLS model; the reported Shapley R² is the mean payout across orderings. <code>mc_standard_error</code> quantifies Monte Carlo uncertainty. The profession SVD dimensions are added together as one predictor, consistent with the fixed-effect variance allocation. In the plot, each interaction term's Shapley contribution is split equally between its component predictors.</p>{shapley_html}{shapley_figure_html}</section>
        </div></body></html>
        """).strip()
    path = report_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    if IMPORT_ERROR is not None:
        print(f"Missing dependencies: {IMPORT_ERROR}", file=sys.stderr)
        return 1
    args = parse_args()
    if args.shapley_permutations < 1:
        raise ValueError("--shapley-permutations must be at least 1")
    collocate_list_paths = args.collocate_list or [
        Path("collocates_gendered.txt"),
        Path("collocates_names.txt"),
    ]
    allowed_collocates = load_allowed_collocates(collocate_list_paths)
    if args.reuse_existing:
        runs = sorted((args.data_dir / "runs").glob("*/*/run_summary.json"))
        # Also accept the original one-method layout for backwards compatibility.
        if not runs:
            runs = sorted((args.data_dir / "runs").glob("*/run_summary.json"))
        run_dirs = [path.parent for path in runs]
        meta = (
            json.loads(runs[0].read_text())["embedding"]
            if runs
            else {"k": args.k, "source": str(args.profession_collocates)}
        )
    else:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        (args.data_dir / "inputs").mkdir(exist_ok=True)
        inputs = discover_inputs(args.results_csv)
        run_dirs = []
        meta = {
            "k": args.k,
            "source": str(args.profession_collocates.resolve()),
            "allowed_collocate_files": [
                str(path.resolve()) for path in collocate_list_paths
            ],
            "allowed_collocate_count": len(allowed_collocates),
            "methods": EMBEDDING_METHODS,
        }
        for method in EMBEDDING_METHODS:
            embedding, method_meta = compute_profession_embeddings(
                args.profession_collocates, args.k, method, allowed_collocates
            )
            method_meta["allowed_collocate_files"] = [
                str(path.resolve()) for path in collocate_list_paths
            ]
            method_meta["allowed_collocate_count"] = len(allowed_collocates)
            (args.data_dir / f"embedding_{method}.csv").write_text(
                embedding.to_csv(index=False), encoding="utf-8"
            )
            (args.data_dir / f"embedding_{method}_metadata.json").write_text(
                json.dumps(method_meta, indent=2), encoding="utf-8"
            )
            run_dirs.extend(
                run_one(
                    p,
                    embedding,
                    method_meta,
                    args.data_dir,
                    args.maxiter,
                    args.starting_fixed_effect_interactions,
                    method,
                    args.shapley_permutations,
                    args.shapley_random_state,
                )
                for p in tqdm(inputs, desc=f"{method} input files")
            )
    if not run_dirs:
        print("No runs found", file=sys.stderr)
        return 1
    report = build_report(run_dirs, args.report_dir, meta)
    print(f"Wrote report to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
