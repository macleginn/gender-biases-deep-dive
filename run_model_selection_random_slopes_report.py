#!/usr/bin/env python3
"""Run random-slope model selection, aggregate results, and build one HTML report.

This replaces the old shell wrapper, comparison script, and report generator for
the random-slope dissertation model-selection workflow.
"""

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
from itertools import combinations
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
    from scipy.stats import chi2
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
FIXED_PREDICTORS = [
    "tense",
    "semantic_role",
    "syntactic_role",
    "valence",
    "dominance",
    "frequency",
    "lex_emb_norm",
]
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
    "C(valence)[T.-val]": "Valence: negative",
    "C(dominance)[T.-dom]": "Dominance: negative",
}
VARIANCE_LABELS = {
    "random_effect_variance_semantic_role": "Semantic role",
    "random_effect_variance_syntactic_role": "Syntactic role",
    "random_effect_variance_valence": "Valence",
    "random_effect_variance_dominance": "Dominance",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fixed_predictors: list[str]
    group_col: str
    random_predictors: list[str]
    pairwise_interactions: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run random-slope backward selection for he_she odds CSVs, aggregate "
            "all generated artifacts, and write a top-level HTML report."
        )
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
        default=Path("model_selection_data"),
        help="Top-level directory for copied inputs, run artifacts, and aggregate data.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("model_selection_report"),
        help="Top-level directory for report.html, report figures, and report tables.",
    )
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip model fitting and rebuild comparisons/report from existing data-dir run artifacts.",
    )
    parser.add_argument(
        "--starting-fixed-effect-interactions",
        action="store_true",
        help=(
            "Include pairwise interactions among fixed effects in the starting model. "
            "By default, the starting model uses main effects only."
        ),
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


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def term(name: str) -> str:
    if name == "frequency":
        return "log_frequency"
    return f"C({name})" if name in CATEGORICAL_COLUMNS else name


def build_random_formula(random_predictors: list[str]) -> str:
    if not random_predictors:
        return "1"
    return "1 + " + " + ".join(term(name) for name in random_predictors)


def hierarchical_fixed_terms(predictors: list[str], pairwise_interactions: bool) -> list[str]:
    main_effects = [term(name) for name in predictors]
    if not pairwise_interactions:
        return main_effects
    interactions = [f"{left}:{right}" for left, right in combinations(main_effects, 2)]
    return main_effects + interactions


def can_drop_term(current_terms: list[str], term_to_drop: str) -> bool:
    if ":" in term_to_drop:
        return True
    return not any(term_to_drop in t.split(":") for t in current_terms if ":" in t)


def safe_run_dir(input_path: Path, runs_dir: Path) -> Path:
    digest = hashlib.sha1(str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return runs_dir / f"{slugify(input_path.stem) or 'input'}__{digest}"


def discover_inputs(paths: list[Path]) -> list[Path]:
    if paths:
        csvs = paths
    else:
        csvs = sorted(Path("modelling_data").glob("he_she_odds_results__*.csv"))
    missing = [path for path in csvs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input CSVs: " + ", ".join(map(str, missing)))
    if not csvs:
        raise FileNotFoundError("No he_she_odds_results__*.csv files found in modelling_data.")
    return [path.resolve() for path in csvs]


def discover_existing_run_dirs(data_dir: Path) -> list[Path]:
    runs_dir = data_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs directory found under {data_dir}")
    run_dirs = sorted(
        path for path in runs_dir.iterdir() if path.is_dir() and (path / "run_summary.json").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(f"No existing run directories with run_summary.json found under {runs_dir}")
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
        raise ValueError(f"{path} contains non-positive frequency values, which cannot be logged.")
    df["log_frequency"] = np.log(df["frequency"])
    return df.dropna(subset=sorted(REQUIRED_COLUMNS)).copy()


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


def extract_random_effect_variances(result: Any) -> dict[str, float]:
    cov_re = pd.DataFrame(
        np.asarray(result.cov_re), index=result.cov_re.index, columns=result.cov_re.columns
    )
    variances: dict[str, float] = {}
    for name in RANDOM_PREDICTORS:
        matches = [label for label in cov_re.index if str(label).startswith(f"C({name})")]
        if matches:
            variances[f"random_effect_variance_{name}"] = float(
                cov_re.loc[matches[0], matches[0]]
            )
    return variances


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
    formula = "log_he_she_odds ~ " + " + ".join(fixed_terms)
    re_formula = build_random_formula(random_predictors)
    fit_errors: list[dict[str, str]] = []

    for method in ["lbfgs", "bfgs", "cg", "powell", "nm"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    formula=formula,
                    data=df,
                    groups=df[group_col],
                    re_formula=re_formula,
                )
                result = model.fit(reml=False, method=method, maxiter=maxiter, disp=False)

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
            metrics = {
                "formula": formula,
                "re_formula": re_formula,
                "group_col": group_col,
                "optimizer": method,
                "converged": bool(getattr(result, "converged", False)),
                "aic": float(result.aic),
                "bic": float(result.bic),
                "log_likelihood": float(result.llf),
                "nobs": int(result.nobs),
                "residual_variance": float(result.scale),
                "R2m": float(r2["R2m"]),
                "R2c": float(r2["R2c"]),
                **extract_random_effect_variances(result),
            }
            (model_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            (model_dir / "summary.txt").write_text(
                result.summary().as_text(), encoding="utf-8"
            )
            spec = ModelSpec(name, [], group_col, random_predictors)
            return {"status": "ok", "spec": spec, "result": result, "metrics": metrics}
        except Exception as exc:  # pragma: no cover - optimizer/runtime dependent
            fit_errors.append({"optimizer": method, "error": repr(exc)})

    spec = ModelSpec(name, [], group_col, random_predictors)
    (model_dir / "fit_failed.json").write_text(
        json.dumps({"status": "failed", "spec": asdict(spec), "fit_errors": fit_errors}, indent=2),
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


def likelihood_ratio_test(full_result: Any, reduced_result: Any) -> dict[str, float | int]:
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


def report_payload(fit: dict[str, Any], selected_terms: list[str], run_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": fit["status"],
        "selected_model": fit["spec"].name,
        "selected_terms": selected_terms,
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
                test = reduced_fit["lr_test"]
                candidate_rows.append(
                    {
                        "step": reduced_fit["step"],
                        "candidate_term": reduced_fit["candidate_term"],
                        "candidate_model": reduced_fit["spec"].name,
                        **test,
                        "aic": float(reduced_fit["metrics"]["aic"]),
                        "bic": float(reduced_fit["metrics"]["bic"]),
                        "log_likelihood": float(reduced_fit["metrics"]["log_likelihood"]),
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


def run_one_input(
    input_csv: Path,
    data_dir: Path,
    *,
    maxiter: int,
    alpha: float,
    starting_fixed_effect_interactions: bool,
) -> Path:
    inputs_dir = data_dir / "inputs"
    runs_dir = data_dir / "runs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_csv, inputs_dir / input_csv.name)

    run_dir = safe_run_dir(input_csv, runs_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    df = load_results_csv(input_csv)
    df.to_csv(run_dir / "prepared_results.csv", index=False)
    (run_dir / "data_profile.json").write_text(
        json.dumps(
            {
                "input_csv": str(input_csv.resolve()),
                "copied_input_csv": str((inputs_dir / input_csv.name).resolve()),
                "rows": int(len(df)),
                "columns": list(df.columns),
                "category_levels": {
                    column: df[column].astype("string").dropna().sort_values().unique().tolist()
                    for column in sorted(CATEGORICAL_COLUMNS)
                    if column in df.columns
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    started = datetime.now(timezone.utc).isoformat()
    selected_fit, trace, selected_terms = select_backward_model(
        df,
        run_dir,
        maxiter=maxiter,
        alpha=alpha,
        starting_fixed_effect_interactions=starting_fixed_effect_interactions,
    )
    selected_payload = report_payload(selected_fit, selected_terms, run_dir)
    baseline_fit = fit_mixed_model_terms(
        df,
        name="selected_mixed_random_slopes__random_intercept_baseline",
        fixed_terms=selected_terms,
        group_col="profession",
        random_predictors=[],
        run_dir=run_dir,
        maxiter=maxiter,
    )
    baseline_payload = report_payload(baseline_fit, selected_terms, run_dir)
    report_reuse_summary = {
        **selected_payload,
        "selected_model_metrics": selected_payload,
        "random_intercept_baseline_metrics": baseline_payload,
        "top_candidates_by_aic_path": (
            str((run_dir / "top_candidates_by_aic.csv").resolve())
            if (run_dir / "top_candidates_by_aic.csv").exists()
            else None
        ),
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
                    "selected_model": "selected_mixed_random_slopes",
                    "status": selected_fit["status"],
                    "selection_steps": int(len(trace)) if not trace.empty else 0,
                    "selected_terms": selected_terms,
                    "report_reuse_summary_path": str(
                        (run_dir / "report_reuse_summary.json").resolve()
                    ),
                    "selected_model_metrics": selected_payload,
                    "random_intercept_baseline_metrics": baseline_payload,
                    "top_candidates_by_aic_path": report_reuse_summary["top_candidates_by_aic_path"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


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
            metrics_path = model_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = load_json(metrics_path)
            rows.append(
                {
                    "run_dir": str(run_dir.resolve()),
                    "run_name": run_dir.name,
                    "input_csv": input_csv,
                    "target_model": target_model,
                    "model_name": model_dir.name,
                    "model_dir": str(model_dir.resolve()),
                    "selection_variant": (
                        "random_intercept_baseline"
                        if model_dir.name.endswith("__random_intercept_baseline")
                        else "random_slopes"
                    ),
                    **metrics,
                }
            )
    df = pd.DataFrame(rows)
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
    expanded = metrics.loc[~metrics["model_name"].str.endswith("__random_intercept_baseline")]
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
    metrics.sort_values(["target_model", "aic", "model_name"], na_position="last").to_csv(
        comparisons_dir / "model_metrics_across_selection_runs.csv", index=False
    )

    if not top_candidates.empty:
        top_candidates.sort_values(
            ["target_model", "aic", "bic", "candidate_model"], na_position="last"
        ).groupby("target_model", as_index=False).head(5).to_csv(
            comparisons_dir / "top_candidates_by_input.csv", index=False
        )
    else:
        top_candidates.to_csv(comparisons_dir / "top_candidates_by_input.csv", index=False)

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
    for _, best in best_models.iterrows():
        fixed = load_fixed_effects(Path(best["model_dir"]))
        for _, row in fixed.iterrows():
            coefficient_rows.append(
                {
                    "target_model": best["target_model"],
                    "best_model_name": best["model_name"],
                    "best_model_aic": best["aic"],
                    **row.to_dict(),
                    **{column: best.get(column) for column in RANDOM_EFFECT_VARIANCE_COLUMNS},
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
    coefficients.to_csv(comparisons_dir / "best_coefficients_by_target_model.csv", index=False)
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
    path.write_text(html, encoding="utf-8")
    return html


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
    top_candidates = artifacts["top_candidates"].copy()
    best_models = artifacts["best_models"].copy()
    coefficients = artifacts["coefficients"].copy()
    random_effects = artifacts["random_effects"].copy()
    for df in [metrics, best_models, coefficients, random_effects]:
        if "target_model" in df.columns:
            df["model"] = df["target_model"].map(clean_model_name)
    if not top_candidates.empty and "target_model" in top_candidates.columns:
        top_candidates["model"] = top_candidates["target_model"].map(clean_model_name)

    model_order = sorted(best_models["model"].unique())
    best_models["selected_model"] = best_models["model_name"].str.replace("_", " ", regex=False)
    fit_table = best_models[
        [
            "model",
            "selected_model",
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
            "selected_model": "Selected model",
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
    fit_html = write_table(fit_table, tab_dir / "model_fit.html")

    if not top_candidates.empty:
        candidate_source = top_candidates.copy()
        candidate_source["candidate_label"] = candidate_source["candidate_model"]
        candidate_source["candidate_detail"] = candidate_source["candidate_term"]
    else:
        candidate_source = (
            metrics.sort_values(["target_model", "aic", "bic", "model_name"])
            .groupby("target_model", as_index=False)
            .head(5)
            .copy()
        )
        candidate_source["candidate_label"] = candidate_source["model_name"]
        candidate_source["candidate_detail"] = candidate_source["selection_variant"]
    candidate_source["model"] = candidate_source["target_model"].map(clean_model_name)
    candidate_source["delta_aic"] = candidate_source.groupby("target_model")["aic"].transform(
        lambda s: s - s.min()
    )
    candidate_table = candidate_source[
        [
            "model",
            "candidate_label",
            "candidate_detail",
            "converged",
            "aic",
            "delta_aic",
            "bic",
            "R2m",
            "R2c",
            "formula",
            "re_formula",
        ]
    ].rename(
        columns={
            "model": "Model",
            "candidate_label": "Candidate",
            "candidate_detail": "Candidate detail",
            "converged": "Converged",
            "aic": "AIC",
            "delta_aic": "Delta AIC",
            "bic": "BIC",
            "R2m": "Marginal R2",
            "R2c": "Conditional R2",
            "formula": "Formula",
            "re_formula": "Random formula",
        }
    )
    candidates_html = write_table(
        candidate_table, tab_dir / "top_model_candidates.html", classes="data-table compact"
    )

    baseline = metrics.loc[
        metrics["model_name"].str.endswith("__random_intercept_baseline")
    ].copy()
    baseline["model"] = baseline["target_model"].map(clean_model_name)
    increment = best_models.merge(
        baseline.sort_values(["target_model", "aic"]).groupby("target_model", as_index=False).first(),
        on="target_model",
        suffixes=("_expanded", "_baseline"),
    )
    increment["Model"] = increment["model_expanded"]
    increment["Delta conditional R2"] = increment["R2c_expanded"] - increment["R2c_baseline"]
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
    increment_html = write_table(increment_table, tab_dir / "random_slope_increment.html")

    baseline_explained = pd.DataFrame(
        {
            "Model": increment["Model"],
            "Fixed effects R2": increment["R2m_baseline"],
            "Random intercept only R2": increment["R2c_baseline"] - increment["R2m_baseline"],
            "Random intercept + fixed effects R2": increment["R2c_baseline"],
        }
    )
    baseline_explained["Fixed-effects share within baseline explained variance"] = safe_divide(
        baseline_explained["Fixed effects R2"],
        baseline_explained["Random intercept + fixed effects R2"],
    )
    baseline_explained["Random-intercept-only share within baseline explained variance"] = safe_divide(
        baseline_explained["Random intercept only R2"],
        baseline_explained["Random intercept + fixed effects R2"],
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
            "Additional random-slope R2": increment["R2c_expanded"] - increment["R2c_baseline"],
            "Full model with random slopes R2": increment["R2c_expanded"],
        }
    )
    expanded_explained["Baseline share within full explained variance"] = safe_divide(
        expanded_explained["Random intercept + fixed effects R2"],
        expanded_explained["Full model with random slopes R2"],
    )
    expanded_explained["Random-slope share within full explained variance"] = safe_divide(
        expanded_explained["Additional random-slope R2"],
        expanded_explained["Full model with random slopes R2"],
    )
    expanded_explained_html = write_table(
        expanded_explained,
        tab_dir / "full_explained_variance_decomposition.html",
        classes="data-table compact",
    )
    combined_explained = pd.DataFrame(
        {
            "Model": increment["Model"],
            "Random intercept only R2": increment["R2c_baseline"] - increment["R2m_baseline"],
            "Fixed effects R2": increment["R2m_baseline"],
            "Additional random-slope R2": increment["R2c_expanded"] - increment["R2c_baseline"],
            "Full model with random slopes R2": increment["R2c_expanded"],
        }
    )

    plt.figure(figsize=(9, 5))
    r2_plot = fit_table.melt(
        id_vars="Model",
        value_vars=["Marginal R2", "Conditional R2"],
        var_name="Statistic",
        value_name="R2",
    )
    sns.barplot(data=r2_plot, x="Model", y="R2", hue="Statistic", order=model_order)
    plt.xticks(rotation=25, ha="right")
    plt.xlabel("")
    savefig(fig_dir / "r2_comparison.png")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=increment_table, x="Model", y="Delta AIC", order=model_order, color="#315c70")
    plt.axhline(0, color="#222", linewidth=1)
    plt.xticks(rotation=25, ha="right")
    plt.xlabel("")
    savefig(fig_dir / "random_slope_increment_aic.png")

    combined_explained_plot = combined_explained.set_index("Model")[
        [
            "Random intercept only R2",
            "Fixed effects R2",
            "Additional random-slope R2",
        ]
    ].rename(
        columns={
            "Random intercept only R2": "Random intercept only",
            "Fixed effects R2": "Fixed effects",
            "Additional random-slope R2": "Additional random slopes",
        }
    )
    combined_explained_plot.loc[model_order].plot(
        kind="bar",
        stacked=True,
        figsize=(9, 5),
        color=["#6c8da6", "#d7a84f", "#b75d42"],
    )
    plt.ylabel("Explained variance (R2)")
    plt.xlabel("")
    plt.ylim(0, max(1.0, float(combined_explained["Full model with random slopes R2"].max()) * 1.08))
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(fig_dir / "explained_variance_decomposition.png")

    variance_table = best_models[["model", *RANDOM_EFFECT_VARIANCE_COLUMNS]].rename(
        columns={"model": "Model", **VARIANCE_LABELS}
    )
    variance_table["Total random-slope variance"] = variance_table[
        list(VARIANCE_LABELS.values())
    ].sum(axis=1)
    variance_html = write_table(variance_table, tab_dir / "variance_decomposition.html")
    variance_prop = variance_table.set_index("Model")[list(VARIANCE_LABELS.values())]
    variance_prop = variance_prop.div(variance_prop.sum(axis=1), axis=0)
    variance_prop.loc[model_order].plot(kind="bar", stacked=True, figsize=(9, 5))
    plt.ylabel("Proportion of random-slope variance")
    plt.xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(fig_dir / "variance_decomposition.png")

    re_model = random_effects.groupby(["model", "Unnamed: 0"], as_index=False)[
        [c for c in RANDOM_EFFECT_LABELS if c in random_effects.columns]
    ].mean()
    heatmap_blocks: list[str] = []
    corr_blocks: list[str] = []
    pca_html = "<p>No random-effect rows were available for PCA.</p>"
    if not re_model.empty and "Group" in re_model.columns:
        profession_order = (
            re_model.groupby("Unnamed: 0")["Group"].mean().sort_values(ascending=False).index
        )
        for column, label in RANDOM_EFFECT_LABELS.items():
            if column not in re_model.columns:
                continue
            name = slugify(label.lower())
            heat = (
                re_model.pivot(index="Unnamed: 0", columns="model", values=column)
                .reindex(index=profession_order, columns=model_order)
            )
            plt.figure(figsize=(9, 6))
            sns.heatmap(heat, center=0, cmap="RdBu_r", cbar_kws={"label": "Random effect"})
            plt.xlabel("")
            plt.ylabel("Profession")
            plt.title(label)
            heat_name = f"profession_{name}_heatmap.png"
            savefig(fig_dir / heat_name)
            heatmap_blocks.append(figure(f"figures/{heat_name}", f"{label} by profession."))

            plt.figure(figsize=(6, 5))
            sns.heatmap(heat.corr(), annot=True, fmt=".2f", center=0, vmin=-1, vmax=1, cmap="RdBu_r")
            plt.title(label)
            corr_name = f"{name}_correlation.png"
            savefig(fig_dir / corr_name)
            corr_blocks.append(figure(f"figures/{corr_name}", f"Cross-model correlation: {label}."))

        features = [c for c in RANDOM_EFFECT_LABELS if c in re_model.columns]
        if len(re_model) >= 3 and len(features) >= 2:
            X = StandardScaler().fit_transform(re_model[features])
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X)
            pca_df = re_model[["model", "Unnamed: 0"]].copy()
            pca_df["PC1"] = pcs[:, 0]
            pca_df["PC2"] = pcs[:, 1]
            plt.figure(figsize=(9, 6))
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="model", s=55)
            for _, row in pca_df.iterrows():
                plt.text(row["PC1"], row["PC2"], str(row["Unnamed: 0"]), fontsize=7)
            plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
            savefig(fig_dir / "profession_pca.png")
            pca_html = figure("figures/profession_pca.png", "PCA of profession random effects.")

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
    coef_html = write_table(coef_table, tab_dir / "coefficients.html", classes="data-table compact")
    coef_heat = coef_model.pivot(index="term", columns="model", values="coef").reindex(columns=model_order)
    plt.figure(figsize=(10, max(5, 0.35 * len(coef_heat))))
    sns.heatmap(coef_heat, center=0, cmap="RdBu_r", cbar_kws={"label": "Coefficient"})
    plt.xlabel("")
    plt.ylabel("Fixed-effect term")
    savefig(fig_dir / "coefficient_heatmap.png")

    cards = [
        ("Target models", str(metrics["target_model"].nunique())),
        ("Professions", str(random_effects["Unnamed: 0"].nunique())),
        ("Observations/model", f"{int(best_models['nobs'].median()):,}"),
        ("Best models converged", "yes" if bool(best_models["converged"].all()) else "no"),
    ]
    cards_html = "\n".join(
        f'<div class="metric-card"><div class="metric-value">{escape(value)}</div>'
        f'<div class="metric-label">{escape(label)}</div></div>'
        for label, value in cards
    )
    html = textwrap.dedent(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Random-Slope Model Selection Report</title>
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
            .table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 14px; margin: 12px 0 18px; background: white; }}
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
              <h1>Random-Slope Model Selection Report</h1>
              <p class="subtitle">
                Backward-selected mixed-effects models for log he/she odds. Positive
                estimates shift predictions toward "he"; negative estimates shift them
                toward "she". Generated on {date.today().isoformat()}.
              </p>
              <div class="metric-grid">{cards_html}</div>
            </header>
            <section>
              <h2>Model Fit</h2>
              <p class="section-note">Best expanded random-slope model per target model.</p>
              <div class="table-wrap">{fit_html}</div>
              <div class="figure-grid">
                {figure("figures/r2_comparison.png", "Marginal and conditional R2.")}
                {figure("figures/random_slope_increment_aic.png", "AIC change versus the random-intercept baseline.")}
              </div>
            </section>
            <section>
              <h2>Top Candidates by AIC</h2>
              <p class="section-note">The five strongest fitted candidate models retained for each input file, ranked by AIC.</p>
              <div class="table-wrap">{candidates_html}</div>
            </section>
            <section>
              <h2>Random-Slope Increment</h2>
              <p class="section-note">
                The baseline fit uses the selected fixed effects with a random intercept only.
                The expanded fit adds the selected random-slope structure on top of that same
                baseline.
              </p>
              <div class="table-wrap">{increment_html}</div>
            </section>
            <section>
              <h2>Explained-Variance Decomposition</h2>
              <p class="section-note">
                The plot shows absolute explained variance (`R2`) split into three stacked
                components: the random intercept alone, fixed effects, and the additional
                contribution from random slopes.
              </p>
              {figure("figures/explained_variance_decomposition.png", "Explained variance split into random intercept only, fixed effects, and additional random slopes.")}
              <div class="table-wrap">{baseline_explained_html}</div>
              <div class="table-wrap">{expanded_explained_html}</div>
            </section>
            <section>
              <h2>Random-Effect Variance</h2>
              <p class="section-note">Baseline rows are the random-intercept models generated in the same run with the selected fixed terms.</p>
              <div class="table-wrap">{variance_html}</div>
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
              <div class="table-wrap">{coef_html}</div>
            </section>
          </div>
        </body>
        </html>
        """
    ).strip()
    report_path = report_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    if IMPORT_ERROR is not None:
        print(dependency_message(IMPORT_ERROR), file=sys.stderr)
        return 1

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "report_dir": str(args.report_dir.resolve()),
        "maxiter": args.maxiter,
        "alpha": args.alpha,
        "reuse_existing": bool(args.reuse_existing),
        "starting_fixed_effect_interactions": bool(args.starting_fixed_effect_interactions),
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
            print(f"Running random-slope selection for {input_csv.name}")
            run_dirs.append(
                run_one_input(
                    input_csv,
                    args.data_dir,
                    maxiter=args.maxiter,
                    alpha=args.alpha,
                    starting_fixed_effect_interactions=args.starting_fixed_effect_interactions,
                )
            )
        manifest["run_dirs"] = [str(path.resolve()) for path in run_dirs]

    (args.data_dir / "execution_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    metrics = collect_model_rows(run_dirs)
    if metrics.empty:
        print("No fitted model metrics were generated.", file=sys.stderr)
        return 1
    top_candidates = collect_top_candidate_rows(run_dirs)
    artifacts = aggregate_outputs(metrics, top_candidates, args.data_dir / "comparisons")
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
