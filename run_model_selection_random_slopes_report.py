#!/usr/bin/env python3
"""Fit full random-slope models, aggregate results, and build one HTML report."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sys
import textwrap
import warnings
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

IMPORT_ERROR: Exception | None = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import statsmodels.formula.api as smf
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from tqdm.auto import tqdm

    log = np.log
except ImportError as exc:  # pragma: no cover - dependency guard
    IMPORT_ERROR = exc


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
RANDOM_PREDICTORS = ["semantic_role", "syntactic_role", "valence", "dominance"]
RANDOM_EFFECT_VARIANCE_COLUMNS = [
    "random_effect_variance_semantic_role",
    "random_effect_variance_syntactic_role",
    "random_effect_variance_valence",
    "random_effect_variance_dominance",
]
RANDOM_EFFECT_LABELS = {
    "Group": "Profession intercept",
    "C(semantic_role)[T.patient]": "Semantic role: patient",
    "C(syntactic_role)[T.subject]": "Syntactic role: subject",
    "C(valence, Treatment(reference='-val'))[T.+val]": "Valence: positive",
    "C(dominance, Treatment(reference='-dom'))[T.+dom]": "Dominance: positive",
}
VARIANCE_LABELS = {
    "random_effect_variance_semantic_role": "Semantic role",
    "random_effect_variance_syntactic_role": "Syntactic role",
    "random_effect_variance_valence": "Valence",
    "random_effect_variance_dominance": "Dominance",
}
NUMERICAL_ISSUE_WARNING_PATTERNS = (
    r"optimization failed to converge",
    r"mixedlm optimization failed",
    r"gradient optimization failed",
    r"the mle may be on the boundary of the parameter space",
    r"hessian matrix .* not positive definite",
    r"random effects covariance is singular",
    r"singular matrix",
    r"invalid value encountered",
    r"overflow encountered",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fixed_predictors: list[str]
    group_col: str
    random_predictors: list[str]
    pairwise_interactions: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit full random-slope models for he/she odds CSVs and write an HTML report."
    )
    parser.add_argument(
        "results_csv",
        type=Path,
        nargs="*",
        help="Input CSVs. Defaults to all modelling_data/he_she_odds_results__*.csv files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("full_model_data"),
        help="Top-level directory for copied inputs, run artifacts, and aggregate data.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("full_model_report"),
        help="Top-level directory for report.html, report figures, and report tables.",
    )
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument(
        "--shapley-permutations",
        type=int,
        default=500,
        help="Random orderings for fixed-effect OLS Shapley R² values (default: 500).",
    )
    parser.add_argument("--shapley-random-state", type=int, default=0)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip model fitting and rebuild the report from existing full-model artifacts.",
    )
    return parser.parse_args()


def dependency_message(exc: Exception) -> str:
    return (
        "Missing required Python dependencies.\n"
        f"Import error: {exc}\n\n"
        "Install dependencies with:\n"
        "  uv sync\n"
    )


def clean_model_name(value: str) -> str:
    value = str(value).replace("he_she_odds_results__", "")
    return re.sub(r"__[0-9a-f]{8}$", "", value)


def load_model_display_config() -> tuple[list[str], dict[str, str]]:
    model_ids = [
        line.strip()
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
    normalized_ids = [model_id.replace("/", "_") for model_id in model_ids]
    return normalized_ids, dict(zip(normalized_ids, model_labels))


def display_model_table(
    table: pd.DataFrame, model_order: list[str], model_labels: dict[str, str]
) -> pd.DataFrame:
    table = table.copy()
    table["_model_order"] = pd.Categorical(
        table["Model"], categories=model_order, ordered=True
    )
    table = table.sort_values("_model_order", kind="stable").drop(
        columns="_model_order"
    )
    table["Model"] = table["Model"].map(model_labels)
    return table


def fixed_effect_label(term_name: str) -> str:
    if term_name == "Intercept":
        return "Intercept"
    label = re.sub(r"C\(([^,]+), Treatment\(reference='[^']+'\)\)", r"\1", term_name)
    label = re.sub(r"C\(([^)]+)\)", r"\1", label)
    label = re.sub(r"\[T\.([^]]+)\]", r" (\1)", label)
    label = label.replace("_", " ")
    return label[:1].upper() + label[1:]


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def term(name: str) -> str:
    if name == "frequency":
        return "log_frequency"
    if name in TREATMENT_REFERENCES:
        return f"C({name}, Treatment(reference='{TREATMENT_REFERENCES[name]}'))"
    return f"C({name})" if name in CATEGORICAL_COLUMNS else name


def build_random_formula(random_predictors: list[str]) -> str:
    if not random_predictors:
        return "1"
    return "1 + " + " + ".join(term(name) for name in random_predictors)


def hierarchical_fixed_terms(
    predictors: list[str], pairwise_interactions: bool = True
) -> list[str]:
    main_effects = [term(name) for name in predictors]
    if not pairwise_interactions:
        return main_effects
    interactions = [
        f"{left}:{right}"
        for index, left in enumerate(main_effects)
        for right in main_effects[index + 1 :]
    ]
    return main_effects + interactions


def can_drop_term(current_terms: list[str], term_to_drop: str) -> bool:
    if ":" in term_to_drop:
        return True
    return not any(term_to_drop in t.split(":") for t in current_terms if ":" in t)


def safe_run_dir(input_path: Path, runs_dir: Path) -> Path:
    digest = hashlib.sha1(str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return runs_dir / f"{slugify(input_path.stem) or 'input'}__{digest}"


def run_dir_has_completed_full_model(run_dir: Path) -> bool:
    if not (
        (run_dir / "run_summary.json").exists()
        and (run_dir / "prepared_results.csv").exists()
        and (run_dir / "models" / "full_mixed_random_slopes" / "metrics.json").exists()
    ):
        return False
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    return (
        summary.get("primary_analysis", {}).get("preprocessing_version")
        == PREPROCESSING_VERSION
    )


def discover_inputs(paths: list[Path]) -> list[Path]:
    if paths:
        csvs = paths
    else:
        csvs = sorted(Path("modelling_data").glob("he_she_odds_results__*.csv"))
    missing = [path for path in csvs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input CSVs: " + ", ".join(map(str, missing)))
    if not csvs:
        raise FileNotFoundError(
            "No he_she_odds_results__*.csv files found in modelling_data."
        )
    return [path.resolve() for path in csvs]


def discover_existing_run_dirs(data_dir: Path) -> list[Path]:
    runs_dir = data_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs directory found under {data_dir}")
    run_dirs = sorted(
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "run_summary.json").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No existing run directories with run_summary.json found under {runs_dir}"
        )
    return run_dirs


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
        raise ValueError(
            f"{path} contains non-positive frequency values, which cannot be logged."
        )
    df["log_frequency"] = np.log(df["frequency"])
    df = df.dropna(subset=sorted(REQUIRED_COLUMNS)).copy()

    # Orthogonalize lexical-embedding norm against log frequency before scaling,
    # so its coefficient represents variation not linearly associated with word
    # frequency.  The residual replaces the original column used in the formula.
    frequency_design = np.column_stack(
        [np.ones(len(df)), df["log_frequency"].to_numpy()]
    )
    lex_embedding = df["lex_emb_norm"].to_numpy()
    frequency_coefficients, *_ = np.linalg.lstsq(
        frequency_design, lex_embedding, rcond=None
    )
    df["lex_emb_norm"] = lex_embedding - frequency_design @ frequency_coefficients

    scaling: dict[str, dict[str, float]] = {}
    for column in NUMERICAL_PREDICTORS:
        mean = float(df[column].mean())
        std = float(df[column].std(ddof=0))
        if not np.isfinite(std) or std == 0:
            raise ValueError(
                f"{path} has no usable variation in {column} for standardization."
            )
        df[column] = (df[column] - mean) / std
        scaling[column] = {"mean": mean, "std": std}
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


def mixedlm_r_squared(result: Any) -> dict[str, float]:
    fixed_design = np.asarray(result.model.exog)
    fixed_beta = np.asarray(result.fe_params)
    var_fixed = float(np.var(fixed_design @ fixed_beta, ddof=1))
    random_design = getattr(result.model, "exog_re", None)
    if random_design is None:
        var_random = 0.0
    else:
        cov_re = np.asarray(result.cov_re)
        var_random = float(
            np.mean(np.einsum("ij,jk,ik->i", random_design, cov_re, random_design))
        )
    var_residual = float(result.scale)
    total = var_fixed + var_random + var_residual
    if total <= 0:
        return {"R2m": float("nan"), "R2c": float("nan")}
    return {"R2m": var_fixed / total, "R2c": (var_fixed + var_random) / total}


def fixed_effect_variances_from_design(
    design: np.ndarray, design_info: Any, beta: pd.Series | np.ndarray
) -> dict[str, float]:
    """Allocate fitted fixed-effect variance, including covariance between terms.

    Each term receives its own fitted-contribution variance plus its covariance
    with every other term.  This splits every pairwise covariance equally between
    its two terms, so the allocations sum exactly to the variance of the full
    fixed-effects linear predictor (apart from floating-point rounding).
    """
    if design_info is None:
        return {}
    beta = np.asarray(beta)
    term_names: list[str] = []
    contributions: list[np.ndarray] = []
    for term_name, column_slice in design_info.term_name_slices.items():
        if term_name == "Intercept":
            continue
        term_names.append(term_name)
        contributions.append(design[:, column_slice] @ beta[column_slice])
    if not contributions:
        return {}
    contribution_matrix = np.column_stack(contributions)
    covariance = np.atleast_2d(np.cov(contribution_matrix, rowvar=False, ddof=1))
    allocations = covariance.sum(axis=1)
    return {
        f"fixed_effect_variance_{term_name}": float(allocation)
        for term_name, allocation in zip(term_names, allocations)
    }


def extract_fixed_effect_variances(result: Any) -> dict[str, float]:
    design_info = getattr(result.model.data, "design_info", None)
    return fixed_effect_variances_from_design(
        np.asarray(result.model.exog), design_info, result.fe_params
    )


def random_effect_variances_from_covariance(cov_re: pd.DataFrame) -> dict[str, float]:
    """Return the random-slope variance for each configured predictor.

    Valence and dominance use treatment-coded formulas, whose covariance labels
    begin with ``C(valence,`` and ``C(dominance,`` rather than ``C(valence)``
    and ``C(dominance)``.  Matching the common ``C(<predictor>`` prefix covers
    both forms.
    """
    variances: dict[str, float] = {}
    for name in RANDOM_PREDICTORS:
        matches = [
            label for label in cov_re.index if str(label).startswith(f"C({name}")
        ]
        if matches:
            variances[f"random_effect_variance_{name}"] = float(
                cov_re.loc[matches[0], matches[0]]
            )
    return variances


def extract_random_effect_variances(result: Any) -> dict[str, float]:
    cov_re = pd.DataFrame(
        np.asarray(result.cov_re),
        index=result.cov_re.index,
        columns=result.cov_re.columns,
    )
    return random_effect_variances_from_covariance(cov_re)


def warning_messages(caught_warnings: list[warnings.WarningMessage]) -> list[str]:
    return [
        f"{warning.category.__name__}: {warning.message}" for warning in caught_warnings
    ]


def has_numerical_issue_warnings(
    caught_warnings: list[warnings.WarningMessage],
) -> bool:
    for warning in caught_warnings:
        warning_text = f"{warning.category.__name__}: {warning.message}".lower()
        if any(
            re.search(pattern, warning_text)
            for pattern in NUMERICAL_ISSUE_WARNING_PATTERNS
        ):
            return True
    return False


def fit_mixed_model_terms(
    df: pd.DataFrame,
    *,
    name: str,
    fixed_terms: list[str],
    group_col: str,
    random_predictors: list[str],
    run_dir: Path,
    maxiter: int,
) -> dict[str, Any]:
    model_dir = run_dir / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    formula = "log_he_she_odds ~ " + (" + ".join(fixed_terms) if fixed_terms else "1")
    re_formula = build_random_formula(random_predictors)
    fit_errors: list[dict[str, str]] = []

    for method in ["lbfgs", "bfgs", "cg", "powell", "nm"]:
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula=formula,
                    data=df,
                    groups=df[group_col],
                    re_formula=re_formula,
                )
                result = model.fit(
                    reml=False, method=method, maxiter=maxiter, disp=False
                )

            warning_texts = warning_messages(caught_warnings)
            numerical_issue_warnings = has_numerical_issue_warnings(caught_warnings)
            fixed_names = list(result.fe_params.index)
            intervals = result.conf_int().loc[fixed_names]
            fixed_effects = pd.DataFrame(
                {
                    "coef": result.fe_params,
                    "std_err": result.bse_fe,
                    "z": result.tvalues.loc[fixed_names],
                    "p_value": result.pvalues.loc[fixed_names],
                    "ci_low": intervals[0],
                    "ci_high": intervals[1],
                }
            )
            fixed_effects.index.name = "term"
            fixed_effects.to_csv(model_dir / "fixed_effects.csv")
            pd.DataFrame(
                np.asarray(result.cov_re),
                index=result.cov_re.index,
                columns=result.cov_re.columns,
            ).to_csv(model_dir / "random_effects_covariance.csv")
            pd.DataFrame.from_dict(result.random_effects, orient="index").to_csv(
                model_dir / "group_random_effects.csv"
            )

            r2 = mixedlm_r_squared(result)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                summary_text = result.summary().as_text()
            metrics = {
                "formula": formula,
                "re_formula": re_formula,
                "group_col": group_col,
                "optimizer": method,
                "converged": bool(getattr(result, "converged", False)),
                "warning_count": len(warning_texts),
                "warnings": warning_texts,
                "numerical_issue_warnings": numerical_issue_warnings,
                "aic": float(result.aic),
                "bic": float(result.bic),
                "log_likelihood": float(result.llf),
                "nobs": int(result.nobs),
                "residual_variance": float(result.scale),
                "R2m": float(r2["R2m"]),
                "R2c": float(r2["R2c"]),
                **extract_fixed_effect_variances(result),
                **extract_random_effect_variances(result),
            }
            (model_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            (model_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
            spec = ModelSpec(name, [], group_col, random_predictors)
            return {"status": "ok", "spec": spec, "result": result, "metrics": metrics}
        except Exception as exc:  # pragma: no cover - optimizer/runtime dependent
            fit_errors.append({"optimizer": method, "error": repr(exc)})

    spec = ModelSpec(name, [], group_col, random_predictors)
    (model_dir / "fit_failed.json").write_text(
        json.dumps(
            {"status": "failed", "spec": asdict(spec), "fit_errors": fit_errors},
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"status": "failed", "spec": spec, "fit_errors": fit_errors}


def evaluate_candidate_fit(
    df: pd.DataFrame,
    *,
    name: str,
    fixed_terms: list[str],
    group_col: str,
    random_predictors: list[str],
    run_dir: Path,
    maxiter: int,
    current_fit: dict[str, Any],
    candidate_term: str,
    step: int,
) -> dict[str, Any]:
    reduced_fit = fit_mixed_model_terms(
        df,
        name=name,
        fixed_terms=fixed_terms,
        group_col=group_col,
        random_predictors=random_predictors,
        run_dir=run_dir,
        maxiter=maxiter,
    )
    if reduced_fit["status"] != "ok":
        return reduced_fit

    test = likelihood_ratio_test(current_fit, reduced_fit)
    return {
        "status": "ok",
        "spec": reduced_fit["spec"],
        "metrics": reduced_fit["metrics"],
        "candidate_model": reduced_fit["spec"].name,
        "candidate_term": candidate_term,
        "step": step,
        "lr_test": test,
        "llf": float(reduced_fit["result"].llf),
        "df_modelwc": int(reduced_fit["result"].df_modelwc),
    }


def _extract_fit_stat(fit: Any, key: str) -> float | int:
    if isinstance(fit, dict):
        if key in fit:
            return fit[key]
        if key == "llf" and "result" in fit:
            return fit["result"].llf
        if key == "df_modelwc" and "result" in fit:
            return fit["result"].df_modelwc
        if "metrics" in fit and key == "llf":
            metrics = fit["metrics"]
            if "log_likelihood" in metrics:
                return metrics["log_likelihood"]
    return getattr(fit, key)


def likelihood_ratio_test(
    full_result: Any, reduced_result: Any
) -> dict[str, float | int]:
    full_llf = float(_extract_fit_stat(full_result, "llf"))
    reduced_llf = float(_extract_fit_stat(reduced_result, "llf"))
    full_df = int(_extract_fit_stat(full_result, "df_modelwc"))
    reduced_df = int(_extract_fit_stat(reduced_result, "df_modelwc"))
    lr_stat = 2.0 * (full_llf - reduced_llf)
    df_diff = int(full_df - reduced_df)
    return {
        "lr_stat": float(lr_stat),
        "df_diff": df_diff,
        "p_value": float(chi2.sf(lr_stat, df_diff)) if df_diff > 0 else float("nan"),
    }


def report_payload(
    fit: dict[str, Any], fixed_terms: list[str], run_dir: Path
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": fit["status"],
        "full_model": fit["spec"].name,
        "fixed_terms": fixed_terms,
    }
    if fit["status"] != "ok":
        return payload

    metrics = dict(fit["metrics"])
    marginal_r2 = float(metrics.get("R2m", float("nan")))
    conditional_r2 = float(metrics.get("R2c", float("nan")))
    random_component_r2 = conditional_r2 - marginal_r2
    model_dir = run_dir / "models" / fit["spec"].name
    payload.update(
        {
            "model_dir": str(model_dir.resolve()),
            "fixed_effects_path": str((model_dir / "fixed_effects.csv").resolve()),
            "group_random_effects_path": str(
                (model_dir / "group_random_effects.csv").resolve()
            ),
            "random_effects_covariance_path": str(
                (model_dir / "random_effects_covariance.csv").resolve()
            ),
            "random_component_R2": random_component_r2,
            "random_component_share_of_R2c": (
                random_component_r2 / conditional_r2 if conditional_r2 else float("nan")
            ),
            "fixed_effect_share_of_R2c": (
                marginal_r2 / conditional_r2 if conditional_r2 else float("nan")
            ),
            **metrics,
        }
    )
    return payload


def select_backward_model(
    df: pd.DataFrame,
    run_dir: Path,
    *,
    maxiter: int,
    alpha: float,
    starting_fixed_effect_interactions: bool,
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    spec = ModelSpec(
        name="selected_mixed_random_slopes",
        fixed_predictors=FIXED_PREDICTORS,
        group_col="profession",
        random_predictors=RANDOM_PREDICTORS,
        pairwise_interactions=starting_fixed_effect_interactions,
    )
    current_terms = hierarchical_fixed_terms(
        spec.fixed_predictors, spec.pairwise_interactions
    )
    current_fit = fit_mixed_model_terms(
        df,
        name=f"{spec.name}__full",
        fixed_terms=current_terms,
        group_col=spec.group_col,
        random_predictors=spec.random_predictors,
        run_dir=run_dir,
        maxiter=maxiter,
    )
    if current_fit["status"] != "ok":
        return current_fit, pd.DataFrame(), current_terms
    current_fit_summary = {
        "llf": float(current_fit["result"].llf),
        "df_modelwc": int(current_fit["result"].df_modelwc),
    }

    trace_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    step = 0
    progress_bar = tqdm(
        total=0,
        desc="Evaluated candidate models",
        unit="model",
        dynamic_ncols=True,
        leave=True,
    )
    try:
        while True:
            candidate_tasks: list[tuple[str, list[str], str]] = []
            for candidate_term in current_terms:
                if not can_drop_term(current_terms, candidate_term):
                    continue
                reduced_terms = [t for t in current_terms if t != candidate_term]
                candidate_tasks.append(
                    (
                        f"{spec.name}__candidate__{step}__{slugify(candidate_term)}",
                        reduced_terms,
                        candidate_term,
                    )
                )

            if not candidate_tasks:
                break

            progress_bar.total += len(candidate_tasks)
            progress_bar.refresh()
            worker_count = min(len(candidate_tasks), os.cpu_count() or 1)
            step_results: list[dict[str, Any]] = []
            if worker_count <= 1:
                for candidate_name, reduced_terms, candidate_term in candidate_tasks:
                    step_results.append(
                        evaluate_candidate_fit(
                            df,
                            name=candidate_name,
                            fixed_terms=reduced_terms,
                            group_col=spec.group_col,
                            random_predictors=spec.random_predictors,
                            run_dir=run_dir,
                            maxiter=maxiter,
                            current_fit=current_fit_summary,
                            candidate_term=candidate_term,
                            step=step,
                        )
                    )
                    progress_bar.update(1)
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                    future_map = {
                        executor.submit(
                            evaluate_candidate_fit,
                            df,
                            name=candidate_name,
                            fixed_terms=reduced_terms,
                            group_col=spec.group_col,
                            random_predictors=spec.random_predictors,
                            run_dir=run_dir,
                            maxiter=maxiter,
                            current_fit=current_fit_summary,
                            candidate_term=candidate_term,
                            step=step,
                        ): candidate_term
                        for candidate_name, reduced_terms, candidate_term in candidate_tasks
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        step_results.append(future.result())
                        progress_bar.update(1)

            candidates: list[tuple[float, str, dict[str, Any]]] = []
            for reduced_fit in step_results:
                if reduced_fit["status"] != "ok":
                    continue
                if reduced_fit["metrics"].get("numerical_issue_warnings"):
                    continue
                test = reduced_fit["lr_test"]
                candidate_rows.append(
                    {
                        "step": reduced_fit["step"],
                        "candidate_term": reduced_fit["candidate_term"],
                        "candidate_model": reduced_fit["spec"].name,
                        **test,
                        "aic": float(reduced_fit["metrics"]["aic"]),
                        "bic": float(reduced_fit["metrics"]["bic"]),
                        "log_likelihood": float(
                            reduced_fit["metrics"]["log_likelihood"]
                        ),
                        "R2m": float(reduced_fit["metrics"]["R2m"]),
                        "R2c": float(reduced_fit["metrics"]["R2c"]),
                        "converged": bool(reduced_fit["metrics"]["converged"]),
                        "nobs": int(reduced_fit["metrics"]["nobs"]),
                        "optimizer": reduced_fit["metrics"]["optimizer"],
                        "formula": reduced_fit["metrics"]["formula"],
                        "re_formula": reduced_fit["metrics"]["re_formula"],
                    }
                )
                trace_rows.append(
                    {
                        "step": reduced_fit["step"],
                        "candidate_term": reduced_fit["candidate_term"],
                        **test,
                        "candidate_model": reduced_fit["spec"].name,
                    }
                )
                candidates.append(
                    (float(test["p_value"]), reduced_fit["candidate_term"], reduced_fit)
                )

            if not candidates:
                break
            best_p, best_term, best_fit = max(candidates, key=lambda item: item[0])
            if np.isnan(best_p) or best_p <= alpha:
                break
            current_terms = [t for t in current_terms if t != best_term]
            current_fit_summary = {
                "llf": float(best_fit["llf"]),
                "df_modelwc": int(best_fit["df_modelwc"]),
            }
            step += 1
    finally:
        progress_bar.close()

    selected_fit = fit_mixed_model_terms(
        df,
        name=spec.name,
        fixed_terms=current_terms,
        group_col=spec.group_col,
        random_predictors=spec.random_predictors,
        run_dir=run_dir,
        maxiter=maxiter,
    )
    trace = pd.DataFrame(trace_rows)
    if not trace.empty:
        trace.to_csv(run_dir / "model_selection_trace.csv", index=False)
    candidate_table = pd.DataFrame(candidate_rows)
    if not candidate_table.empty:
        candidate_table["aic"] = pd.to_numeric(candidate_table["aic"], errors="coerce")
        candidate_table = candidate_table.sort_values(
            ["aic", "bic", "candidate_model"], na_position="last"
        )
        top_candidates = candidate_table.head(5).copy()
        top_candidates.to_csv(run_dir / "top_candidates_by_aic.csv", index=False)
        (run_dir / "top_candidates_by_aic.json").write_text(
            json.dumps(top_candidates.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
    (run_dir / "selected_model.json").write_text(
        json.dumps(
            {
                "selected_model": spec.name,
                "selected_terms": current_terms,
                "alpha": alpha,
                "full_model": asdict(spec),
                "starting_fixed_effect_interactions": starting_fixed_effect_interactions,
                "top_candidates_by_aic_path": (
                    str((run_dir / "top_candidates_by_aic.csv").resolve())
                    if not candidate_table.empty
                    else None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return selected_fit, trace, current_terms


def _shapley_marginal_batch(
    df: pd.DataFrame, terms: list[str], permutations: int, seed: int
) -> np.ndarray:
    """Compute one independent, process-local batch of Shapley orderings."""
    cache: dict[tuple[int, ...], float] = {(): 0.0}

    def utility(indices: tuple[int, ...]) -> float:
        if indices not in cache:
            formula = "log_he_she_odds ~ " + " + ".join(
                terms[index] for index in indices
            )
            cache[indices] = float(smf.ols(formula=formula, data=df).fit().rsquared)
        return cache[indices]

    rng = np.random.default_rng(seed)
    marginal = np.empty((permutations, len(terms)))
    for sample in range(permutations):
        included: tuple[int, ...] = ()
        previous = utility(included)
        for index in rng.permutation(len(terms)):
            included = tuple(sorted((*included, int(index))))
            current = utility(included)
            marginal[sample, index] = current - previous
            previous = current
    return marginal


def decompose_fixed_effect_shapley_r_squared(
    df: pd.DataFrame,
    terms: list[str],
    run_dir: Path,
    permutations: int,
    random_state: int,
) -> pd.DataFrame:
    """Approximate fixed-effect Shapley values with OLS R² as the payout."""
    if permutations < 1:
        raise ValueError("shapley permutations must be at least 1")
    columns = [
        "predictor",
        "shapley_r2",
        "mc_standard_error",
        "relative_r2",
        "permutations",
        "random_state",
        "full_r2",
        "utility",
    ]
    if not terms:
        return pd.DataFrame(columns=columns)

    worker_count = min(permutations, os.cpu_count() or 1)
    batch_count = min(permutations, worker_count * 4)
    batch_sizes = [permutations // batch_count] * batch_count
    for index in range(permutations % batch_count):
        batch_sizes[index] += 1
    seeds = np.random.SeedSequence(random_state).generate_state(batch_count)
    batches = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_shapley_marginal_batch, df, terms, size, int(seed))
            for size, seed in zip(batch_sizes, seeds)
        ]
        with tqdm(
            total=permutations,
            desc=f"Shapley R²: {run_dir.name}",
            unit="ordering",
            dynamic_ncols=True,
            leave=True,
        ) as progress:
            for future in concurrent.futures.as_completed(futures):
                batch = future.result()
                batches.append(batch)
                progress.update(len(batch))
    marginal = np.vstack(batches)
    values = marginal.mean(axis=0)
    full_r2 = float(
        smf.ols(formula="log_he_she_odds ~ " + " + ".join(terms), data=df)
        .fit()
        .rsquared
    )
    result = pd.DataFrame(
        {
            "predictor": terms,
            "shapley_r2": values,
            "mc_standard_error": (
                marginal.std(axis=0, ddof=1) / np.sqrt(permutations)
                if permutations > 1
                else np.zeros(len(terms))
            ),
            "relative_r2": values / full_r2 if full_r2 > 0 else 0.0,
            "permutations": permutations,
            "random_state": random_state,
            "full_r2": full_r2,
            "utility": "OLS fixed-effect R2",
        }
    ).sort_values("shapley_r2", ascending=False, ignore_index=True)
    result.to_csv(run_dir / "shapley_r2_decomposition.csv", index=False)
    return result


def fold_shapley_interactions(decomposition: pd.DataFrame) -> pd.DataFrame:
    """Split interaction Shapley values equally across their simple effects."""
    columns = ["target_model", "predictor", "folded_shapley_r2"]
    if decomposition.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
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
    return (
        pd.DataFrame(rows)
        .groupby(["target_model", "predictor"], as_index=False)["folded_shapley_r2"]
        .sum()
    )


def run_one_input(
    input_csv: Path,
    data_dir: Path,
    *,
    maxiter: int,
    shapley_permutations: int,
    shapley_random_state: int,
) -> Path:
    inputs_dir = data_dir / "inputs"
    runs_dir = data_dir / "runs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_dir = safe_run_dir(input_csv, runs_dir)
    if run_dir_has_completed_full_model(run_dir):
        shapley_path = run_dir / "shapley_r2_decomposition.csv"
        prepared_path = run_dir / "prepared_results.csv"
        if not shapley_path.exists() and prepared_path.exists():
            decompose_fixed_effect_shapley_r_squared(
                pd.read_csv(prepared_path),
                hierarchical_fixed_terms(FIXED_PREDICTORS),
                run_dir,
                shapley_permutations,
                shapley_random_state,
            )
        print(f"Reusing existing full-model run for {input_csv.name}: {run_dir}")
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_csv, inputs_dir / input_csv.name)
    df = load_results_csv(input_csv)
    df.to_csv(run_dir / "prepared_results.csv", index=False)
    (run_dir / "data_profile.json").write_text(
        json.dumps(
            {
                "input_csv": str(input_csv.resolve()),
                "copied_input_csv": str((inputs_dir / input_csv.name).resolve()),
                "rows": int(len(df)),
                "columns": list(df.columns),
                "preprocessing": df.attrs["preprocessing"],
                "category_levels": {
                    column: df[column]
                    .astype("string")
                    .dropna()
                    .sort_values()
                    .unique()
                    .tolist()
                    for column in sorted(CATEGORICAL_COLUMNS)
                    if column in df.columns
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    started = datetime.now(timezone.utc).isoformat()
    fixed_terms = hierarchical_fixed_terms(FIXED_PREDICTORS)
    full_fit = fit_mixed_model_terms(
        df,
        name="full_mixed_random_slopes",
        fixed_terms=fixed_terms,
        group_col="profession",
        random_predictors=RANDOM_PREDICTORS,
        run_dir=run_dir,
        maxiter=maxiter,
    )
    full_payload = report_payload(full_fit, fixed_terms, run_dir)
    decompose_fixed_effect_shapley_r_squared(
        df, fixed_terms, run_dir, shapley_permutations, shapley_random_state
    )
    baseline_fit = fit_mixed_model_terms(
        df,
        name="full_mixed_random_slopes__random_intercept_baseline",
        fixed_terms=fixed_terms,
        group_col="profession",
        random_predictors=[],
        run_dir=run_dir,
        maxiter=maxiter,
    )
    baseline_payload = report_payload(baseline_fit, fixed_terms, run_dir)
    standalone_random_intercept_fit = fit_mixed_model_terms(
        df,
        name="random_intercept_only",
        fixed_terms=[],
        group_col="profession",
        random_predictors=[],
        run_dir=run_dir,
        maxiter=maxiter,
    )
    standalone_random_intercept_payload = report_payload(
        standalone_random_intercept_fit, [], run_dir
    )
    report_reuse_summary = {
        **full_payload,
        "full_model_metrics": full_payload,
        "random_intercept_baseline_metrics": baseline_payload,
        "standalone_random_intercept_metrics": standalone_random_intercept_payload,
    }
    (run_dir / "report_reuse_summary.json").write_text(
        json.dumps(report_reuse_summary, indent=2), encoding="utf-8"
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "started_at_utc": started,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_csv": str(input_csv.resolve()),
                "primary_analysis": {
                    "full_model": "full_mixed_random_slopes",
                    "status": full_fit["status"],
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "fixed_terms": fixed_terms,
                    "report_reuse_summary_path": str(
                        (run_dir / "report_reuse_summary.json").resolve()
                    ),
                    "full_model_metrics": full_payload,
                    "random_intercept_baseline_metrics": baseline_payload,
                    "standalone_random_intercept_metrics": standalone_random_intercept_payload,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def ensure_shapley_results(
    run_dirs: list[Path], permutations: int, random_state: int
) -> None:
    """Create missing per-run Shapley files while preserving precomputed ones."""
    terms = hierarchical_fixed_terms(FIXED_PREDICTORS)
    for run_dir in run_dirs:
        shapley_path = run_dir / "shapley_r2_decomposition.csv"
        prepared_path = run_dir / "prepared_results.csv"
        if not shapley_path.exists() and prepared_path.exists():
            decompose_fixed_effect_shapley_r_squared(
                pd.read_csv(prepared_path),
                terms,
                run_dir,
                permutations,
                random_state,
            )


def ensure_standalone_random_intercept_results(
    run_dirs: list[Path], maxiter: int
) -> None:
    for run_dir in run_dirs:
        model_dir = run_dir / "models" / "random_intercept_only"
        metrics_path = model_dir / "metrics.json"
        prepared_path = run_dir / "prepared_results.csv"
        if metrics_path.exists() or not prepared_path.exists():
            continue
        fit_mixed_model_terms(
            pd.read_csv(prepared_path),
            name="random_intercept_only",
            fixed_terms=[],
            group_col="profession",
            random_predictors=[],
            run_dir=run_dir,
            maxiter=maxiter,
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def target_model_from_input(input_csv: str | None, fallback: str) -> str:
    stem = Path(input_csv).stem if input_csv else fallback
    return clean_model_name(stem.split("__", 1)[1] if "__" in stem else stem)


def collect_model_rows(run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_summary = load_json(run_dir / "run_summary.json")
        input_csv = run_summary.get("input_csv")
        target_model = target_model_from_input(input_csv, run_dir.name)
        for model_dir in sorted((run_dir / "models").iterdir()):
            if model_dir.name not in {
                "full_mixed_random_slopes",
                "full_mixed_random_slopes__random_intercept_baseline",
                "random_intercept_only",
            }:
                continue
            metrics_path = model_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = load_json(metrics_path)
            covariance_path = model_dir / "random_effects_covariance.csv"
            if covariance_path.exists():
                covariance = pd.read_csv(covariance_path, index_col=0)
                recovered_variances = random_effect_variances_from_covariance(
                    covariance
                )
                metrics = {**recovered_variances, **metrics}
            rows.append(
                {
                    "run_dir": str(run_dir.resolve()),
                    "run_name": run_dir.name,
                    "input_csv": input_csv,
                    "target_model": target_model,
                    "model_name": model_dir.name,
                    "model_dir": str(model_dir.resolve()),
                    "fit_variant": (
                        "standalone_random_intercept"
                        if model_dir.name == "random_intercept_only"
                        else (
                            "random_intercept_baseline"
                            if model_dir.name.endswith("__random_intercept_baseline")
                            else "random_slopes"
                        )
                    ),
                    **metrics,
                }
            )
    df = pd.DataFrame(rows)
    for column in RANDOM_EFFECT_VARIANCE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    for column in [
        "aic",
        "bic",
        "log_likelihood",
        "residual_variance",
        "R2m",
        "R2c",
        "nobs",
        *RANDOM_EFFECT_VARIANCE_COLUMNS,
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in [
        column for column in df.columns if column.startswith("fixed_effect_variance_")
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "converged" in df.columns:
        df["converged"] = df["converged"].astype("boolean")
    return df


def collect_top_candidate_rows(run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        top_candidates_path = run_dir / "top_candidates_by_aic.csv"
        if not top_candidates_path.exists():
            continue
        run_summary = load_json(run_dir / "run_summary.json")
        input_csv = run_summary.get("input_csv")
        target_model = target_model_from_input(input_csv, run_dir.name)
        candidates = pd.read_csv(top_candidates_path)
        if candidates.empty:
            continue
        candidates["aic"] = pd.to_numeric(candidates["aic"], errors="coerce")
        for _, row in candidates.iterrows():
            rows.append(
                {
                    "run_dir": str(run_dir.resolve()),
                    "run_name": run_dir.name,
                    "input_csv": input_csv,
                    "target_model": target_model,
                    **row.to_dict(),
                }
            )
    df = pd.DataFrame(rows)
    for column in ["aic", "bic", "log_likelihood", "nobs"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "converged" in df.columns:
        df["converged"] = df["converged"].astype("boolean")
    return df


def load_fixed_effects(model_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(model_dir / "fixed_effects.csv")
    for column in ["coef", "std_err", "p_value", "ci_low", "ci_high"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_group_random_effects(model_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(model_dir / "group_random_effects.csv")
    if "Unnamed: 0" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Unnamed: 0"})
    return df


def best_expanded_models(metrics: pd.DataFrame) -> pd.DataFrame:
    expanded = metrics.loc[metrics["model_name"].eq("full_mixed_random_slopes")]
    return (
        expanded.dropna(subset=["aic"])
        .sort_values(["target_model", "aic", "bic", "model_name"], na_position="last")
        .groupby("target_model", as_index=False)
        .first()
    )


def aggregate_outputs(
    metrics: pd.DataFrame, top_candidates: pd.DataFrame, comparisons_dir: Path
) -> dict[str, pd.DataFrame]:
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    metrics.sort_values(
        ["target_model", "aic", "model_name"], na_position="last"
    ).to_csv(comparisons_dir / "model_metrics_across_full_model_runs.csv", index=False)

    summary = (
        metrics.groupby(["target_model", "model_name"], dropna=False)
        .agg(
            runs=("run_name", "nunique"),
            mean_aic=("aic", "mean"),
            mean_bic=("bic", "mean"),
            mean_R2m=("R2m", "mean"),
            mean_R2c=("R2c", "mean"),
            converged_rate=("converged", "mean"),
            **{column: (column, "mean") for column in RANDOM_EFFECT_VARIANCE_COLUMNS},
        )
        .reset_index()
        .sort_values(["target_model", "mean_aic", "mean_bic"], na_position="last")
    )
    summary.to_csv(comparisons_dir / "model_summary.csv", index=False)

    best_models = best_expanded_models(metrics)
    coefficient_rows: list[dict[str, Any]] = []
    random_effect_rows: list[dict[str, Any]] = []
    for best_index, best in best_models.iterrows():
        fixed = load_fixed_effects(Path(best["model_dir"]))
        for _, row in fixed.iterrows():
            coefficient_rows.append(
                {
                    "target_model": best["target_model"],
                    "best_model_name": best["model_name"],
                    "best_model_aic": best["aic"],
                    **row.to_dict(),
                    **{
                        column: best.get(column)
                        for column in RANDOM_EFFECT_VARIANCE_COLUMNS
                    },
                }
            )
        random_effects = load_group_random_effects(Path(best["model_dir"]))
        for _, row in random_effects.iterrows():
            random_effect_rows.append(
                {
                    "target_model": best["target_model"],
                    "best_model_name": best["model_name"],
                    "best_model_aic": best["aic"],
                    **row.to_dict(),
                }
            )

    coefficients = pd.DataFrame(coefficient_rows)
    random_effects = pd.DataFrame(random_effect_rows)
    coefficients.to_csv(
        comparisons_dir / "best_coefficients_by_target_model.csv", index=False
    )
    random_effects.to_csv(
        comparisons_dir / "best_random_effects_by_target_model.csv", index=False
    )
    best_models.to_csv(comparisons_dir / "best_models_by_target_model.csv", index=False)
    return {
        "metrics": metrics,
        "top_candidates": top_candidates,
        "summary": summary,
        "best_models": best_models,
        "coefficients": coefficients,
        "random_effects": random_effects,
    }


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def write_table(df: pd.DataFrame, path: Path, *, classes: str = "data-table") -> str:
    html = df.to_html(
        index=False,
        border=0,
        classes=classes,
        escape=True,
        float_format=lambda x: f"{x:.3f}",
    )
    collapsed_html = (
        '<details class="table-details"><summary>Show table</summary>'
        f'<div class="table-wrap">{html}</div></details>'
    )
    path.write_text(collapsed_html, encoding="utf-8")
    return collapsed_html


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def figure(src: str, caption: str) -> str:
    return (
        f'<figure><img src="{escape(src)}" alt="{escape(caption)}" loading="lazy">'
        f"<figcaption>{escape(caption)}</figcaption></figure>"
    )


def build_report(artifacts: dict[str, pd.DataFrame], report_dir: Path) -> Path:
    sns.set_theme(style="whitegrid", context="notebook")
    fig_dir = report_dir / "figures"
    tab_dir = report_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    metrics = artifacts["metrics"].copy()
    best_models = artifacts["best_models"].copy()
    coefficients = artifacts["coefficients"].copy()
    random_effects = artifacts["random_effects"].copy()
    for df in [metrics, best_models, coefficients, random_effects]:
        if "target_model" in df.columns:
            df["model"] = df["target_model"].map(clean_model_name)

    available_models = set(best_models["model"].unique())
    configured_model_order, configured_model_labels = load_model_display_config()
    model_order = [
        model for model in configured_model_order if model in available_models
    ]
    model_order.extend(sorted(available_models - set(model_order)))
    plot_model_labels = {
        model: configured_model_labels.get(model, model) for model in model_order
    }
    plot_model_order = [plot_model_labels[model] for model in model_order]
    plot_model_order_reversed = plot_model_order[::-1]
    shapley_frames = []
    for _, model in best_models.iterrows():
        path = Path(model["run_dir"]) / "shapley_r2_decomposition.csv"
        if path.exists():
            shapley = pd.read_csv(path)
            shapley.insert(0, "target_model", model["target_model"])
            shapley_frames.append(shapley)
    shapley_decomposition = (
        pd.concat(shapley_frames, ignore_index=True)
        if shapley_frames
        else pd.DataFrame()
    )
    shapley_decomposition.to_csv(
        report_dir / "shapley_r2_decomposition.csv", index=False
    )
    folded_shapley = fold_shapley_interactions(shapley_decomposition)
    folded_shapley.to_csv(report_dir / "shapley_r2_folded.csv", index=False)
    best_models["fitted_model"] = best_models["model_name"].str.replace(
        "_", " ", regex=False
    )
    fit_table = best_models[
        [
            "model",
            "fitted_model",
            "converged",
            "optimizer",
            "nobs",
            "aic",
            "bic",
            "log_likelihood",
            "R2m",
            "R2c",
            "residual_variance",
        ]
    ].rename(
        columns={
            "model": "Model",
            "fitted_model": "Fitted model",
            "converged": "Converged",
            "optimizer": "Optimizer",
            "nobs": "N",
            "aic": "AIC",
            "bic": "BIC",
            "log_likelihood": "Log likelihood",
            "R2m": "Marginal R2",
            "R2c": "Conditional R2",
            "residual_variance": "Residual variance",
        }
    )
    fit_table = display_model_table(fit_table, model_order, plot_model_labels)
    fit_html = write_table(fit_table, tab_dir / "model_fit.html")
    shapley_html = "<p>No Monte Carlo Shapley R² results.</p>"
    shapley_figure_html = ""
    if not folded_shapley.empty:
        shapley_table = folded_shapley.copy()
        shapley_table["Model"] = shapley_table["target_model"].map(clean_model_name)
        shapley_table = shapley_table.drop(columns="target_model")
        shapley_table["Model"] = shapley_table["Model"].map(plot_model_labels)
        shapley_table["predictor"] = shapley_table["predictor"].map(fixed_effect_label)
        shapley_table = shapley_table.rename(
            columns={"predictor": "Predictor", "folded_shapley_r2": "Shapley R2"}
        )
        shapley_html = write_table(
            shapley_table,
            tab_dir / "shapley_r2_decomposition.html",
            classes="data-table compact",
        )
        shapley_plot = folded_shapley.copy()
        shapley_plot["Model"] = shapley_plot["target_model"].map(clean_model_name)
        shapley_plot["Model"] = shapley_plot["Model"].map(plot_model_labels)
        shapley_plot["Predictor"] = shapley_plot["predictor"].map(fixed_effect_label)
        shapley_plot = shapley_plot.pivot_table(
            index="Model",
            columns="Predictor",
            values="folded_shapley_r2",
            aggfunc="sum",
            fill_value=0.0,
        ).reindex(plot_model_order_reversed, fill_value=0.0)
        predictor_order = shapley_plot.sum(axis=0).sort_values(ascending=False).index
        shapley_plot = shapley_plot.reindex(columns=predictor_order)
        shapley_colors = sns.color_palette("husl", n_colors=len(predictor_order))
        ax = shapley_plot.plot(
            kind="barh",
            stacked=True,
            figsize=(9, max(4.5, len(plot_model_order_reversed) * 0.55 + 1.5)),
            color=shapley_colors,
            width=0.9,
        )
        ax.set_xlabel("Shapley R² contribution (interactions split equally)")
        ax.set_ylabel("")
        ax.axvline(0, color="#222", linewidth=1)
        ax.legend(title="Predictor", bbox_to_anchor=(1.02, 1), loc="upper left")
        savefig(fig_dir / "shapley_r2_decomposition.png")
        shapley_figure_html = figure(
            "figures/shapley_r2_decomposition.png",
            "Monte Carlo Shapley R² contributions by fixed-effect predictor and model.",
        )

    baseline = metrics.loc[
        metrics["model_name"].str.endswith("__random_intercept_baseline")
    ].copy()
    baseline["model"] = baseline["target_model"].map(clean_model_name)
    standalone_random_intercept = metrics.loc[
        metrics["model_name"].eq("random_intercept_only")
    ].copy()
    standalone_random_intercept["model"] = standalone_random_intercept[
        "target_model"
    ].map(clean_model_name)
    increment = best_models.merge(
        baseline.sort_values(["target_model", "aic"])
        .groupby("target_model", as_index=False)
        .first(),
        on="target_model",
        suffixes=("_expanded", "_baseline"),
    )
    increment = increment.merge(
        standalone_random_intercept[["target_model", "R2c"]].rename(
            columns={"R2c": "R2c_standalone_random_intercept"}
        ),
        on="target_model",
        how="left",
    )
    increment["Model"] = increment["model_expanded"]
    increment["Delta conditional R2"] = (
        increment["R2c_expanded"] - increment["R2c_baseline"]
    )
    increment["Delta AIC"] = increment["aic_expanded"] - increment["aic_baseline"]
    increment_table = increment[
        [
            "Model",
            "R2c_baseline",
            "R2c_expanded",
            "Delta conditional R2",
            "aic_baseline",
            "aic_expanded",
            "Delta AIC",
            "re_formula_baseline",
            "re_formula_expanded",
        ]
    ].rename(
        columns={
            "R2c_baseline": "Baseline conditional R2",
            "R2c_expanded": "Expanded conditional R2",
            "aic_baseline": "Baseline AIC",
            "aic_expanded": "Expanded AIC",
            "re_formula_baseline": "Baseline random effects",
            "re_formula_expanded": "Expanded random effects",
        }
    )
    increment_table = display_model_table(
        increment_table, model_order, plot_model_labels
    )
    increment_html = write_table(
        increment_table, tab_dir / "random_slope_increment.html"
    )

    baseline_explained = pd.DataFrame(
        {
            "Model": increment["Model"],
            "Fixed effects R2": increment["R2m_baseline"],
            "Standalone random intercept R2": increment[
                "R2c_standalone_random_intercept"
            ],
            "Random intercept + fixed effects R2": increment["R2c_baseline"],
        }
    )
    baseline_explained["Fixed-effects share within baseline explained variance"] = (
        safe_divide(
            baseline_explained["Fixed effects R2"],
            baseline_explained["Random intercept + fixed effects R2"],
        )
    )
    baseline_explained[
        "Random-intercept-only share within baseline explained variance"
    ] = safe_divide(
        baseline_explained["Standalone random intercept R2"],
        baseline_explained["Random intercept + fixed effects R2"],
    )
    baseline_explained = display_model_table(
        baseline_explained, model_order, plot_model_labels
    )
    baseline_explained_html = write_table(
        baseline_explained,
        tab_dir / "baseline_explained_variance_decomposition.html",
        classes="data-table compact",
    )

    expanded_explained = pd.DataFrame(
        {
            "Model": increment["Model"],
            "Random intercept + fixed effects R2": increment["R2c_baseline"],
            "Additional random-slope R2": increment["R2c_expanded"]
            - increment["R2c_baseline"],
            "Full model with random slopes R2": increment["R2c_expanded"],
        }
    )
    expanded_explained["Baseline share within full explained variance"] = safe_divide(
        expanded_explained["Random intercept + fixed effects R2"],
        expanded_explained["Full model with random slopes R2"],
    )
    expanded_explained["Random-slope share within full explained variance"] = (
        safe_divide(
            expanded_explained["Additional random-slope R2"],
            expanded_explained["Full model with random slopes R2"],
        )
    )
    expanded_explained = display_model_table(
        expanded_explained, model_order, plot_model_labels
    )
    expanded_explained_html = write_table(
        expanded_explained,
        tab_dir / "full_explained_variance_decomposition.html",
        classes="data-table compact",
    )
    combined_explained = pd.DataFrame(
        {
            "Model": increment["Model"],
            "Standalone random intercept R2": increment[
                "R2c_standalone_random_intercept"
            ],
            "Fixed effects R2": increment["R2m_baseline"],
            "Additional random-slope R2": increment["R2c_expanded"]
            - increment["R2c_baseline"],
            "Full model with random slopes R2": increment["R2c_expanded"],
        }
    )

    plt.figure(figsize=(8, max(3.5, len(model_order) * 0.4 + 1.5)))
    r2_plot = fit_table.melt(
        id_vars="Model",
        value_vars=["Marginal R2", "Conditional R2"],
        var_name="Statistic",
        value_name="R2",
    )
    sns.barplot(
        data=r2_plot, x="R2", y="Model", hue="Statistic", order=plot_model_order
    )
    plt.ylabel("")
    savefig(fig_dir / "r2_comparison.png")

    plt.figure(figsize=(8, max(3.5, len(model_order) * 0.4 + 1.5)))
    increment_plot = increment_table.copy()
    sns.barplot(
        data=increment_plot,
        x="Delta AIC",
        y="Model",
        order=plot_model_order,
        color="#315c70",
    )
    plt.axvline(0, color="#222", linewidth=1)
    plt.ylabel("")
    savefig(fig_dir / "random_slope_increment_aic.png")

    combined_explained_plot = combined_explained.set_index("Model")[
        [
            "Standalone random intercept R2",
            "Fixed effects R2",
            "Additional random-slope R2",
        ]
    ].rename(
        columns={
            "Standalone random intercept R2": "Standalone random intercept",
            "Fixed effects R2": "Fixed effects",
            "Additional random-slope R2": "Additional random slopes",
        }
    )
    combined_explained_plot.index = combined_explained_plot.index.map(plot_model_labels)
    combined_explained_plot.loc[plot_model_order_reversed].plot(
        kind="barh",
        stacked=True,
        width=0.82,
        figsize=(8, max(3.5, len(model_order) * 0.4 + 1.5)),
        color=["#6c8da6", "#d7a84f", "#b75d42"],
    )
    plt.xlabel("Explained variance (R2)")
    plt.ylabel("")
    plt.xlim(
        0,
        max(
            1.0,
            float(combined_explained["Full model with random slopes R2"].max()) * 1.08,
        ),
    )
    plt.legend(title="", bbox_to_anchor=(0.5, -0.16), loc="upper center", ncol=3)
    plt.subplots_adjust(bottom=0.2)
    savefig(fig_dir / "explained_variance_decomposition.png")

    variance_table = best_models[["model", *RANDOM_EFFECT_VARIANCE_COLUMNS]].rename(
        columns={"model": "Model", **VARIANCE_LABELS}
    )
    variance_table["Total random-slope variance"] = variance_table[
        list(VARIANCE_LABELS.values())
    ].sum(axis=1)
    variance_prop = variance_table.set_index("Model")[list(VARIANCE_LABELS.values())]
    variance_prop = variance_prop.div(variance_prop.sum(axis=1), axis=0)
    variance_table = display_model_table(variance_table, model_order, plot_model_labels)
    variance_html = write_table(variance_table, tab_dir / "variance_decomposition.html")
    variance_prop.index = variance_prop.index.map(plot_model_labels)
    variance_prop.loc[plot_model_order_reversed].plot(
        kind="barh", stacked=True, figsize=(8, max(3.5, len(model_order) * 0.4 + 1.5))
    )
    plt.xlabel("Proportion of random-slope variance")
    plt.ylabel("")
    plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(fig_dir / "variance_decomposition.png")

    fixed_variance_columns = sorted(
        column
        for column in best_models.columns
        if column.startswith("fixed_effect_variance_")
    )
    fixed_variance_terms = [
        column.removeprefix("fixed_effect_variance_")
        for column in fixed_variance_columns
    ]
    fixed_variance_labels = {
        column: fixed_effect_label(term_name)
        for column, term_name in zip(fixed_variance_columns, fixed_variance_terms)
    }
    fixed_variance_table = best_models[["model", *fixed_variance_columns]].rename(
        columns={"model": "Model", **fixed_variance_labels}
    )
    fixed_variance_table["Total fixed-effect fitted variance"] = fixed_variance_table[
        list(fixed_variance_labels.values())
    ].sum(axis=1)
    fixed_variance_allocation = fixed_variance_table.set_index("Model")[
        list(fixed_variance_labels.values())
    ]
    fixed_variance_table = display_model_table(
        fixed_variance_table, model_order, plot_model_labels
    )
    fixed_variance_html = write_table(
        fixed_variance_table,
        tab_dir / "fixed_effect_variance.html",
        classes="data-table compact",
    )
    fixed_variance_allocation.index = fixed_variance_allocation.index.map(
        plot_model_labels
    )
    fixed_effect_names = list(fixed_variance_labels.values())
    fixed_effect_numbers = list(range(1, len(fixed_effect_names) + 1))
    # HUSL creates a distinct colour for every full-model term.
    fixed_effect_colors = sns.color_palette("husl", n_colors=len(fixed_effect_names))
    fig, ax = plt.subplots(figsize=(11, max(4.5, len(model_order) * 0.55 + 1.5)))
    fixed_variance_allocation.loc[plot_model_order_reversed].plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=fixed_effect_colors,
    )
    ax.set_xlabel("Covariance-adjusted fixed-effect variance allocation")
    ax.set_ylabel("")

    # Number segments with sufficient room for a legible label; every effect is
    # numbered in the legend below, so very small segments remain identifiable
    # without overlapping labels obscuring the bars.
    for effect_number, container in zip(fixed_effect_numbers, ax.containers):
        ax.bar_label(
            container,
            labels=[
                (
                    str(effect_number)
                    if np.isfinite(bar.get_width()) and abs(bar.get_width()) >= 0.025
                    else ""
                )
                for bar in container
            ],
            label_type="center",
            color="#111111",
            fontsize=8,
            fontweight="bold",
        )
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [
            f"{number}. {name}"
            for number, name in zip(fixed_effect_numbers, fixed_effect_names)
        ],
        title="Fixed effect",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncols=min(3, len(fixed_effect_names)),
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.3)
    savefig(fig_dir / "fixed_effect_variance.png")

    re_model = random_effects.groupby(["model", "Unnamed: 0"], as_index=False)[
        [c for c in RANDOM_EFFECT_LABELS if c in random_effects.columns]
    ].mean()
    heatmap_blocks: list[str] = []
    corr_blocks: list[str] = []
    pca_html = "<p>No random-effect rows were available for PCA.</p>"
    if not re_model.empty and "Group" in re_model.columns:
        profession_order = (
            re_model.groupby("Unnamed: 0")["Group"]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        for column, label in RANDOM_EFFECT_LABELS.items():
            if column not in re_model.columns:
                continue
            name = slugify(label.lower())
            heat = re_model.pivot(
                index="Unnamed: 0", columns="model", values=column
            ).reindex(index=profession_order, columns=model_order)
            heat.columns = heat.columns.map(plot_model_labels)
            plt.figure(figsize=(9, max(6, len(profession_order) * 0.22 + 1.5)))
            heat_ax = sns.heatmap(
                heat, center=0, cmap="RdBu_r", cbar_kws={"label": "Random effect"}
            )
            heat_ax.tick_params(
                axis="y", labelsize=max(4, min(8, 200 / max(1, len(profession_order))))
            )
            plt.xlabel("")
            plt.ylabel("Profession")
            plt.title(label)
            heat_name = f"profession_{name}_heatmap.png"
            savefig(fig_dir / heat_name)
            heatmap_blocks.append(
                figure(f"figures/{heat_name}", f"{label} by profession.")
            )

            correlation_size = 2 * max(8, len(model_order) * 0.55)
            plt.figure(figsize=(correlation_size, correlation_size))
            sns.heatmap(
                heat.corr(),
                annot=True,
                annot_kws={"fontsize": 9},
                fmt=".2f",
                center=0,
                vmin=-1,
                vmax=1,
                cmap="RdBu_r",
            )
            plt.title(label)
            corr_name = f"{name}_correlation.png"
            savefig(fig_dir / corr_name)
            corr_blocks.append(
                figure(f"figures/{corr_name}", f"Cross-model correlation: {label}.")
            )

        features = [c for c in RANDOM_EFFECT_LABELS if c in re_model.columns]
        if len(re_model) >= 3 and len(features) >= 2:
            X = StandardScaler().fit_transform(re_model[features])
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X)
            pca_df = re_model[["model", "Unnamed: 0"]].copy()
            pca_df["model"] = pca_df["model"].map(plot_model_labels)
            pca_df["PC1"] = pcs[:, 0]
            pca_df["PC2"] = pcs[:, 1]
            plt.figure(figsize=(9, 6))
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="model", s=55)
            for _, row in pca_df.iterrows():
                plt.text(row["PC1"], row["PC2"], str(row["Unnamed: 0"]), fontsize=7)
            plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
            savefig(fig_dir / "profession_pca.png")
            pca_html = figure(
                "figures/profession_pca.png", "PCA of profession random effects."
            )

    coef_model = coefficients.groupby(["model", "term"], as_index=False)[
        ["coef", "std_err", "p_value", "ci_low", "ci_high"]
    ].mean()
    coef_table = coef_model.rename(
        columns={
            "model": "Model",
            "term": "Term",
            "coef": "Estimate",
            "std_err": "Std. error",
            "p_value": "p value",
            "ci_low": "CI low",
            "ci_high": "CI high",
        }
    )
    coef_table = display_model_table(coef_table, model_order, plot_model_labels)
    coef_html = write_table(
        coef_table, tab_dir / "coefficients.html", classes="data-table compact"
    )
    coef_heat = coef_model.pivot(index="term", columns="model", values="coef").reindex(
        columns=model_order
    )
    coef_heat.columns = coef_heat.columns.map(plot_model_labels)
    plt.figure(figsize=(10, max(5, 0.35 * len(coef_heat))))
    sns.heatmap(coef_heat, center=0, cmap="RdBu_r", cbar_kws={"label": "Coefficient"})
    plt.xlabel("")
    plt.ylabel("Fixed-effect term")
    savefig(fig_dir / "coefficient_heatmap.png")

    cards = [
        ("Target models", str(metrics["target_model"].nunique())),
        ("Professions", str(random_effects["Unnamed: 0"].nunique())),
        ("Observations/model", f"{int(best_models['nobs'].median()):,}"),
        (
            "Best models converged",
            "yes" if bool(best_models["converged"].all()) else "no",
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
            <title>Full Random-Slope Model Report</title>
          <style>
            :root {{
              --bg: #f4efe6; --panel: #fffaf1; --ink: #1d2625;
              --muted: #64706b; --border: #d9c8ad; --accent: #315c70;
            }}
            body {{
              margin: 0; color: var(--ink);
              font-family: Charter, "Bitstream Charter", Georgia, serif;
              background: radial-gradient(circle at 8% 0%, #ecd0bc, transparent 28rem),
                          radial-gradient(circle at 95% 10%, #c9dce2, transparent 26rem),
                          var(--bg);
            }}
            .page {{ max-width: 1220px; margin: 0 auto; padding: 32px 20px 60px; }}
            header, section {{
              background: rgba(255,250,241,.93); border: 1px solid var(--border);
              border-radius: 24px; padding: 26px; margin-bottom: 22px;
            }}
            h1 {{ font-size: clamp(2.2rem, 5vw, 4.5rem); line-height: .96; margin: 0 0 12px; }}
            h2 {{ margin: 0 0 12px; font-size: 1.7rem; }}
            .subtitle, .section-note, figcaption {{ color: var(--muted); line-height: 1.5; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 20px; }}
            .metric-card {{ border: 1px solid var(--border); border-radius: 16px; padding: 14px; background: #fffdf8; }}
            .metric-value {{ font-size: 1.7rem; font-weight: 700; color: var(--accent); }}
            .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
            figure {{ margin: 16px auto; }}
            img {{ width: 100%; border: 1px solid var(--border); border-radius: 14px; background: white; }}
            .table-details {{ border: 1px solid var(--border); border-radius: 14px; margin: 12px 0 18px; background: #fffdf8; }}
            .table-details summary {{ cursor: pointer; padding: 10px 14px; color: var(--accent); font-weight: 700; }}
            .table-details[open] summary {{ border-bottom: 1px solid var(--border); }}
            .table-wrap {{ overflow-x: auto; background: white; }}
            .data-table {{ width: 100%; border-collapse: collapse; font-size: .92rem; }}
            .data-table th, .data-table td {{ border: 1px solid #e2d4bf; padding: 8px 10px; text-align: left; vertical-align: top; }}
            .data-table th {{ background: #f0dfc7; }}
            .compact {{ font-size: .84rem; }}
            @media (max-width: 760px) {{ .page {{ padding: 14px 10px 36px; }} header, section {{ padding: 16px; }} .figure-grid {{ grid-template-columns: 1fr; }} }}
          </style>
        </head>
        <body>
          <div class="page">
            <header>
              <h1>Full Random-Slope Model Report</h1>
              <p class="subtitle">
                Full mixed-effects models for log he/she odds. Positive
                estimates shift predictions toward "he"; negative estimates shift them
                toward "she". `log_frequency` and the frequency-residualized
                `lex_emb_norm` are standardized before fitting. Generated on
                {date.today().isoformat()}.
              </p>
              <div class="metric-grid">{cards_html}</div>
            </header>
            <section>
              <h2>Model Fit</h2>
              <p class="section-note">Full random-slope model per target model.</p>
              {fit_html}
              <div class="figure-grid">
                {figure("figures/r2_comparison.png", "Marginal and conditional R2.")}
                {figure("figures/random_slope_increment_aic.png", "AIC change versus the random-intercept baseline.")}
              </div>
            </section>
            <section>
              <h2>Random-Slope Increment</h2>
              <p class="section-note">
                The baseline fit uses the full fixed-effects formula with a random intercept only.
                The full fit adds the random-slope structure on top of that same
                baseline.
              </p>
              {increment_html}
            </section>
            <section>
              <h2>Explained-Variance Decomposition</h2>
              <p class="section-note">
                The plot shows absolute explained variance (`R2`) split into three stacked
                                standalone components: the random intercept from a model with no fixed
                                predictors, fixed effects, and the additional contribution from random slopes.
                                These standalone quantities need not sum to the expanded model R2.
              </p>
                            {figure("figures/explained_variance_decomposition.png", "Standalone explained variance from the random intercept, fixed effects, and additional random slopes.")}
              {baseline_explained_html}
              {expanded_explained_html}
            </section>
            <section>
              <h2>Fixed-Effect Variance</h2>
              <p class="section-note">
                Each fixed effect is summarized by the variance of its fitted contribution
                across observations, with each covariance divided equally between its two
                terms. The allocations sum to the variance of the complete fixed-effects
                predictor; negative allocations can occur when a term covaries negatively
                with the others.
              </p>
              {fixed_variance_html}
              {figure("figures/fixed_effect_variance.png", "Covariance-adjusted fixed-effect variance allocation by model term; segment numbers map to the legend below the plot.")}
            </section>
            <section>
              <h2>Random-Effect Variance</h2>
                            <p class="section-note">The random-intercept baseline uses the same full fixed-effects formula; standalone random-intercept importance is reported separately in the explained-variance decomposition.</p>
              {variance_html}
              {figure("figures/variance_decomposition.png", "Random-slope variance decomposition.")}
            </section>
            <section>
              <h2>Profession Structure</h2>
              {''.join(heatmap_blocks)}
            </section>
            <section>
              <h2>Cross-Model Similarity</h2>
              <div class="figure-grid">{''.join(corr_blocks)}</div>
            </section>
            <section>
              <h2>Profession PCA</h2>
              {pca_html}
            </section>
            <section>
              <h2>Fixed Effects</h2>
              {figure("figures/coefficient_heatmap.png", "Fixed-effect coefficients by model.")}
              {coef_html}
            </section>
            <section>
              <h2>Monte Carlo Shapley R² Attribution</h2>
              <p class="section-note">The payout is the R² from an OLS model containing the selected fixed effects. Each term receives its mean R² increase across random term orderings; <code>mc_standard_error</code> quantifies Monte Carlo uncertainty. This fixed-effect OLS utility is reported separately from the mixed-model marginal and conditional R² values above.</p>
              {shapley_figure_html}
              {shapley_html}
            </section>
          </div>
        </body>
        </html>
        """).strip()
    report_path = report_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    if IMPORT_ERROR is not None:
        print(dependency_message(IMPORT_ERROR), file=sys.stderr)
        return 1
    if args.shapley_permutations < 1:
        raise ValueError("--shapley-permutations must be at least 1")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "report_dir": str(args.report_dir.resolve()),
        "maxiter": args.maxiter,
        "shapley_permutations": args.shapley_permutations,
        "shapley_random_state": args.shapley_random_state,
        "reuse_existing": bool(args.reuse_existing),
    }

    run_dirs: list[Path]
    inputs: list[Path] = []
    if args.reuse_existing:
        try:
            run_dirs = discover_existing_run_dirs(args.data_dir)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        manifest["run_dirs"] = [str(path.resolve()) for path in run_dirs]
    else:
        try:
            inputs = discover_inputs(args.results_csv)
        except (FileNotFoundError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        manifest["inputs"] = [str(path) for path in inputs]
        run_dirs = []
        for input_csv in inputs:
            print(f"Fitting full random-slope model for {input_csv.name}")
            run_dirs.append(
                run_one_input(
                    input_csv,
                    args.data_dir,
                    maxiter=args.maxiter,
                    shapley_permutations=args.shapley_permutations,
                    shapley_random_state=args.shapley_random_state,
                )
            )
        manifest["run_dirs"] = [str(path.resolve()) for path in run_dirs]

    (args.data_dir / "execution_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    ensure_shapley_results(
        run_dirs, args.shapley_permutations, args.shapley_random_state
    )
    ensure_standalone_random_intercept_results(run_dirs, args.maxiter)

    metrics = collect_model_rows(run_dirs)
    if metrics.empty:
        print("No fitted model metrics were generated.", file=sys.stderr)
        return 1
    artifacts = aggregate_outputs(
        metrics, pd.DataFrame(), args.data_dir / "comparisons"
    )
    report_path = build_report(artifacts, args.report_dir)

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["report_html"] = str(report_path.resolve())
    (args.data_dir / "execution_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote data artifacts to {args.data_dir}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
