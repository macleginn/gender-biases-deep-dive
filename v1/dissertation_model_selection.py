#!/usr/bin/env python3
"""Mixed-effects analysis with backward selection from a full model.

This follows the same input/output shape as dissertation_analysing_results.py,
but instead of fitting a fixed menu of models, it starts from a full model and
uses likelihood-ratio tests to drop terms while respecting hierarchy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

IMPORT_ERROR: Exception | None = None

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2
    import statsmodels.formula.api as smf
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        tqdm = None  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency guard
    IMPORT_ERROR = exc


REQUIRED_COLUMNS = {
    "tense",
    "syntactic_role",
    "semantic_role",
    "valence",
    "dominance",
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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fixed_predictors: list[str]
    group_col: str
    random_predictors: list[str]
    pairwise_interactions: bool = True


def progress_iterable(iterable: Any, *, total: int | None, desc: str, unit: str) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run mixed-effects analysis with backward selection from a full model."
        )
    )
    parser.add_argument("results_csv", type=Path, help="Path to the input results CSV.")
    parser.add_argument(
        "--profession-metadata-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns 'profession' and 'female_percentage'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis_outputs"),
        help="Root directory for per-input analysis folders. Default: analysis_outputs",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum optimiser iterations per model fit. Default: 1000",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="P-value threshold for backward elimination. Default: 0.05",
    )
    return parser.parse_args()


def dependency_message(exc: Exception) -> str:
    return (
        "Missing required Python dependencies.\n"
        f"Import error: {exc}\n\n"
        "Install the required packages, for example:\n"
        "  uv add numpy pandas scipy statsmodels tqdm\n"
        "or:\n"
        "  pip install numpy pandas scipy statsmodels tqdm"
    )


def sanitized_run_dir(input_path: Path, output_root: Path) -> Path:
    digest = hashlib.sha1(str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", input_path.stem).strip("._") or "input"
    return output_root / "model_selection" / f"{safe_stem}__{digest}"


def term(name: str) -> str:
    return f"C({name})" if name in CATEGORICAL_COLUMNS else name


def build_fixed_formula(predictors: list[str], pairwise_interactions: bool) -> str:
    terms = [term(name) for name in predictors]
    if not pairwise_interactions:
        return " + ".join(terms)
    parts = list(terms)
    parts.extend(f"{left}:{right}" for left, right in combinations(terms, 2))
    return " + ".join(parts)


def build_fixed_formula_from_terms(terms: list[str]) -> str:
    return " + ".join(terms)


def build_random_formula(random_predictors: list[str]) -> str:
    if not random_predictors:
        return "1"
    random_terms = [term(name) for name in random_predictors]
    return "1 + " + " + ".join(random_terms)


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")


def load_results_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ensure_required_columns(df)
    for column in REQUIRED_COLUMNS:
        if column in CATEGORICAL_COLUMNS:
            df[column] = df[column].astype("string").str.strip().astype("category")
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=sorted(REQUIRED_COLUMNS)).copy()


def mixedlm_r_squared(result: Any) -> dict[str, float]:
    fixed_design = np.asarray(result.model.exog)
    fixed_beta = np.asarray(result.fe_params)
    fixed_linear = fixed_design @ fixed_beta
    var_fixed = float(np.var(fixed_linear, ddof=1))
    random_design = getattr(result.model, "exog_re", None)
    if random_design is None:
        var_random = 0.0
    else:
        cov_re = np.asarray(result.cov_re)
        per_row_random_var = np.einsum("ij,jk,ik->i", random_design, cov_re, random_design)
        var_random = float(np.mean(per_row_random_var))
    var_residual = float(result.scale)
    total = var_fixed + var_random + var_residual
    if total <= 0:
        return {"R2m": float("nan"), "R2c": float("nan")}
    return {"R2m": var_fixed / total, "R2c": (var_fixed + var_random) / total}


def fit_mixed_model(
    df: pd.DataFrame,
    spec: ModelSpec,
    output_dir: Path,
    maxiter: int,
) -> dict[str, Any]:
    model_dir = output_dir / "models" / spec.name
    model_dir.mkdir(parents=True, exist_ok=True)
    formula = "log_he_she_odds ~ " + build_fixed_formula(spec.fixed_predictors, spec.pairwise_interactions)
    re_formula = build_random_formula(spec.random_predictors)
    fit_errors: list[dict[str, str]] = []
    for method in ["lbfgs", "bfgs", "cg", "powell", "nm"]:
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula=formula,
                    data=df,
                    groups=df[spec.group_col],
                    re_formula=re_formula,
                )
                result = model.fit(reml=False, method=method, maxiter=maxiter, disp=False)
            r2 = mixedlm_r_squared(result)
            fixed_names = list(result.fe_params.index)
            confidence_intervals = result.conf_int().loc[fixed_names]
            fixed_effects = pd.DataFrame(
                {
                    "coef": result.fe_params,
                    "std_err": result.bse_fe,
                    "z": result.tvalues.loc[fixed_names],
                    "p_value": result.pvalues.loc[fixed_names],
                    "ci_low": confidence_intervals[0],
                    "ci_high": confidence_intervals[1],
                }
            )
            fixed_effects.index.name = "term"
            pd.DataFrame(np.asarray(result.cov_re), index=result.cov_re.index, columns=result.cov_re.columns).to_csv(
                model_dir / "random_effects_covariance.csv"
            )
            pd.DataFrame.from_dict(result.random_effects, orient="index").to_csv(
                model_dir / "group_random_effects.csv"
            )
            fixed_effects.to_csv(model_dir / "fixed_effects.csv")
            metrics = {
                "formula": formula,
                "re_formula": re_formula,
                "group_col": spec.group_col,
                "optimizer": method,
                "converged": bool(getattr(result, "converged", False)),
                "aic": float(result.aic),
                "bic": float(result.bic),
                "log_likelihood": float(result.llf),
                "nobs": int(result.nobs),
                "residual_variance": float(result.scale),
                "R2m": float(r2["R2m"]),
                "R2c": float(r2["R2c"]),
            }
            with (model_dir / "metrics.json").open("w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)
            with (model_dir / "summary.txt").open("w", encoding="utf-8") as fh:
                fh.write(result.summary().as_text())
            return {"status": "ok", "spec": spec, "result": result, "metrics": metrics}
        except Exception as exc:  # pragma: no cover - optimizer/runtime dependent
            fit_errors.append({"optimizer": method, "error": repr(exc)})
    with (model_dir / "fit_failed.json").open("w", encoding="utf-8") as fh:
        json.dump({"status": "failed", "spec": asdict(spec), "fit_errors": fit_errors}, fh, indent=2)
    return {"status": "failed", "spec": spec, "fit_errors": fit_errors}


def fit_mixed_model_terms(
    df: pd.DataFrame,
    *,
    name: str,
    fixed_terms: list[str],
    group_col: str,
    random_predictors: list[str],
    output_dir: Path,
    maxiter: int,
) -> dict[str, Any]:
    spec = ModelSpec(
        name=name,
        fixed_predictors=[],
        group_col=group_col,
        random_predictors=random_predictors,
        pairwise_interactions=False,
    )
    model_dir = output_dir / "models" / spec.name
    model_dir.mkdir(parents=True, exist_ok=True)
    formula = "log_he_she_odds ~ " + build_fixed_formula_from_terms(fixed_terms)
    re_formula = build_random_formula(spec.random_predictors)
    fit_errors: list[dict[str, str]] = []
    for method in ["lbfgs", "bfgs", "cg", "powell", "nm"]:
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula=formula,
                    data=df,
                    groups=df[spec.group_col],
                    re_formula=re_formula,
                )
                result = model.fit(reml=False, method=method, maxiter=maxiter, disp=False)
            r2 = mixedlm_r_squared(result)
            fixed_names = list(result.fe_params.index)
            confidence_intervals = result.conf_int().loc[fixed_names]
            fixed_effects = pd.DataFrame(
                {
                    "coef": result.fe_params,
                    "std_err": result.bse_fe,
                    "z": result.tvalues.loc[fixed_names],
                    "p_value": result.pvalues.loc[fixed_names],
                    "ci_low": confidence_intervals[0],
                    "ci_high": confidence_intervals[1],
                }
            )
            fixed_effects.index.name = "term"
            pd.DataFrame(np.asarray(result.cov_re), index=result.cov_re.index, columns=result.cov_re.columns).to_csv(
                model_dir / "random_effects_covariance.csv"
            )
            pd.DataFrame.from_dict(result.random_effects, orient="index").to_csv(
                model_dir / "group_random_effects.csv"
            )
            fixed_effects.to_csv(model_dir / "fixed_effects.csv")
            metrics = {
                "formula": formula,
                "re_formula": re_formula,
                "group_col": spec.group_col,
                "optimizer": method,
                "converged": bool(getattr(result, "converged", False)),
                "aic": float(result.aic),
                "bic": float(result.bic),
                "log_likelihood": float(result.llf),
                "nobs": int(result.nobs),
                "residual_variance": float(result.scale),
                "R2m": float(r2["R2m"]),
                "R2c": float(r2["R2c"]),
            }
            with (model_dir / "metrics.json").open("w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)
            with (model_dir / "summary.txt").open("w", encoding="utf-8") as fh:
                fh.write(result.summary().as_text())
            return {"status": "ok", "spec": spec, "result": result, "metrics": metrics}
        except Exception as exc:  # pragma: no cover - optimizer/runtime dependent
            fit_errors.append({"optimizer": method, "error": repr(exc)})
    with (model_dir / "fit_failed.json").open("w", encoding="utf-8") as fh:
        json.dump({"status": "failed", "spec": asdict(spec), "fit_errors": fit_errors}, fh, indent=2)
    return {"status": "failed", "spec": spec, "fit_errors": fit_errors}


def likelihood_ratio_test(full_result: Any, reduced_result: Any, full_name: str, reduced_name: str) -> dict[str, float | int | str]:
    lr_stat = 2.0 * (full_result.llf - reduced_result.llf)
    df_diff = int(full_result.df_modelwc - reduced_result.df_modelwc)
    return {
        "full_model": full_name,
        "reduced_model": reduced_name,
        "lr_stat": float(lr_stat),
        "df_diff": df_diff,
        "p_value": float(chi2.sf(lr_stat, df_diff)) if df_diff > 0 else float("nan"),
    }


def hierarchical_fixed_terms(predictors: list[str]) -> list[str]:
    main_effects = [term(name) for name in predictors]
    interactions = [f"{left}:{right}" for left, right in combinations(main_effects, 2)]
    return main_effects + interactions


def build_report_reuse_payload(
    fit: dict[str, Any],
    *,
    selected_terms: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    if fit["status"] != "ok":
        return {
            "status": fit["status"],
            "selected_model": fit["spec"].name,
            "selected_terms": selected_terms,
        }

    metrics = dict(fit["metrics"])
    conditional_r2 = float(metrics.get("R2c", float("nan")))
    marginal_r2 = float(metrics.get("R2m", float("nan")))
    random_component_r2 = conditional_r2 - marginal_r2
    selected_model_dir = output_dir / "models" / fit["spec"].name
    return {
        "status": fit["status"],
        "selected_model": fit["spec"].name,
        "selected_terms": selected_terms,
        "model_dir": str(selected_model_dir.resolve()),
        "fixed_effects_path": str((selected_model_dir / "fixed_effects.csv").resolve()),
        "group_random_effects_path": str((selected_model_dir / "group_random_effects.csv").resolve()),
        "random_effects_covariance_path": str(
            (selected_model_dir / "random_effects_covariance.csv").resolve()
        ),
        "formula": metrics.get("formula"),
        "re_formula": metrics.get("re_formula"),
        "group_col": metrics.get("group_col"),
        "optimizer": metrics.get("optimizer"),
        "converged": metrics.get("converged"),
        "aic": metrics.get("aic"),
        "bic": metrics.get("bic"),
        "log_likelihood": metrics.get("log_likelihood"),
        "nobs": metrics.get("nobs"),
        "residual_variance": metrics.get("residual_variance"),
        "R2m": marginal_r2,
        "R2c": conditional_r2,
        "random_component_R2": random_component_r2,
        "random_component_share_of_R2c": (
            random_component_r2 / conditional_r2 if conditional_r2 else float("nan")
        ),
        "fixed_effect_share_of_R2c": (
            marginal_r2 / conditional_r2 if conditional_r2 else float("nan")
        ),
    }


def can_drop_term(current_terms: list[str], term_to_drop: str) -> bool:
    if ":" not in term_to_drop:
        return not any(
            term_to_drop in interaction.split(":")
            for interaction in current_terms
            if ":" in interaction
        )
    return True


def select_backward_model(
    df: pd.DataFrame,
    spec: ModelSpec,
    output_dir: Path,
    maxiter: int,
    alpha: float,
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    current_terms = hierarchical_fixed_terms(spec.fixed_predictors)
    current_fit = fit_mixed_model_terms(
        df,
        name=f"{spec.name}__full",
        fixed_terms=current_terms,
        group_col=spec.group_col,
        random_predictors=spec.random_predictors,
        output_dir=output_dir,
        maxiter=maxiter,
    )
    if current_fit["status"] != "ok":
        return current_fit, pd.DataFrame(), current_terms

    trace_rows: list[dict[str, Any]] = []
    step = 0
    while True:
        candidates: list[tuple[float, str, dict[str, Any], Any, Any]] = []
        for term_name in current_terms:
            if not can_drop_term(current_terms, term_name):
                continue
            reduced_terms = [t for t in current_terms if t != term_name]
            reduced_fit = fit_mixed_model_terms(
                df,
                name=f"{spec.name}__candidate__{step}__{re.sub(r'[^A-Za-z0-9._-]+', '_', term_name)}",
                fixed_terms=reduced_terms,
                group_col=spec.group_col,
                random_predictors=spec.random_predictors,
                output_dir=output_dir,
                maxiter=maxiter,
            )
            if reduced_fit["status"] != "ok":
                continue
            test = likelihood_ratio_test(
                current_fit["result"],
                reduced_fit["result"],
                f"{spec.name}__current",
                reduced_fit["spec"].name,
            )
            candidates.append((float(test["p_value"]), term_name, test, reduced_fit))
            trace_rows.append(
                {
                    "step": step,
                    "current_model": f"{spec.name}__current",
                    "candidate_term": term_name,
                    **test,
                    "candidate_model": reduced_fit["spec"].name,
                }
            )

        if not candidates:
            break
        best_p, best_term, best_test, best_fit = max(candidates, key=lambda item: item[0])
        if np.isnan(best_p) or best_p <= alpha:
            break

        current_terms = [t for t in current_terms if t != best_term]
        current_fit = best_fit
        step += 1

    selected_fit = fit_mixed_model_terms(
        df,
        name=spec.name,
        fixed_terms=current_terms,
        group_col=spec.group_col,
        random_predictors=spec.random_predictors,
        output_dir=output_dir,
        maxiter=maxiter,
    )
    trace = pd.DataFrame(trace_rows)
    if not trace.empty:
        trace.to_csv(output_dir / "model_selection_trace.csv", index=False)
    with (output_dir / "selected_model.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "selected_model": spec.name,
                "selected_terms": current_terms,
                "alpha": alpha,
                "full_model": asdict(spec),
            },
            fh,
            indent=2,
        )
    return selected_fit, trace, current_terms


def run_primary_analysis(df: pd.DataFrame, output_dir: Path, maxiter: int, alpha: float) -> dict[str, Any]:
    spec = ModelSpec(
        name="selected_mixed",
        fixed_predictors=["tense", "syntactic_role", "valence", "dominance", "semantic_role"],
        group_col="profession",
        random_predictors=["semantic_role", "syntactic_role"],
    )
    fit, trace, selected_terms = select_backward_model(
        df, spec, output_dir, maxiter=maxiter, alpha=alpha
    )
    random_intercept_fit = fit_mixed_model_terms(
        df,
        name=f"{spec.name}__random_intercept_baseline",
        fixed_terms=selected_terms,
        group_col=spec.group_col,
        random_predictors=[],
        output_dir=output_dir,
        maxiter=maxiter,
    )
    selected_model_payload = build_report_reuse_payload(
        fit,
        selected_terms=selected_terms,
        output_dir=output_dir,
    )
    random_intercept_baseline_payload = build_report_reuse_payload(
        random_intercept_fit,
        selected_terms=selected_terms,
        output_dir=output_dir,
    )
    report_reuse_summary = {
        "selected_model_metrics": selected_model_payload,
        "random_intercept_baseline_metrics": random_intercept_baseline_payload,
    }
    with (output_dir / "report_reuse_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(report_reuse_summary, fh, indent=2)
    return {
        "selected_model": spec.name,
        "status": fit["status"],
        "selection_steps": int(len(trace)) if not trace.empty else 0,
        "selected_terms": selected_terms,
        "report_reuse_summary_path": str((output_dir / "report_reuse_summary.json").resolve()),
        "selected_model_metrics": selected_model_payload,
        "random_intercept_baseline_metrics": random_intercept_baseline_payload,
    }


def main() -> int:
    args = parse_args()
    if IMPORT_ERROR is not None:
        print(dependency_message(IMPORT_ERROR), file=sys.stderr)
        return 1
    if not args.results_csv.exists():
        print(f"Results CSV not found: {args.results_csv}", file=sys.stderr)
        return 1

    output_dir = sanitized_run_dir(args.results_csv, args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results_csv(args.results_csv)
    df.to_csv(output_dir / "prepared_results.csv", index=False)
    with (output_dir / "data_profile.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "input_csv": str(args.results_csv.resolve()),
                "rows": int(len(df)),
                "columns": list(df.columns),
                "category_levels": {
                    column: df[column].astype("string").dropna().sort_values().unique().tolist()
                    for column in sorted(CATEGORICAL_COLUMNS)
                    if column in df.columns
                },
            },
            fh,
            indent=2,
        )

    run_summary: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(args.results_csv.resolve()),
        "primary_analysis": run_primary_analysis(df, output_dir, maxiter=args.maxiter, alpha=args.alpha),
    }
    run_summary["metadata_analysis"] = {
        "status": "skipped",
        "reason": "Metadata selection is not implemented in this script.",
    }
    run_summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(run_summary, fh, indent=2)
    print(f"Analysis complete. Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
