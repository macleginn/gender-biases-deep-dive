#!/usr/bin/env python3
"""Standalone analysis pipeline matching dissertation_analysing_results.R.

The script expects a CSV equivalent of the processed results table and writes all
artifacts into an input-specific output directory.
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
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold
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
OPTIONAL_NUMERIC_COLUMNS = {"he_prob", "she_prob", "he_she_odds_ratio"}
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
            "Run the dissertation mixed-effects analysis on a results CSV and "
            "write all outputs into an input-specific folder."
        )
    )
    parser.add_argument("results_csv", type=Path, help="Path to the input results CSV.")
    parser.add_argument(
        "--profession-metadata-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with columns 'profession' and 'female_percentage' to "
            "run the Winobias-style comparison models."
        ),
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
    return parser.parse_args()


def dependency_message(exc: Exception) -> str:
    return (
        "Missing required Python dependencies.\n"
        f"Import error: {exc}\n\n"
        "Install the required packages, for example:\n"
        "  uv add numpy pandas scipy statsmodels scikit-learn\n"
        "or:\n"
        "  pip install numpy pandas scipy statsmodels scikit-learn"
    )


def sanitized_run_dir(input_path: Path, output_root: Path) -> Path:
    digest = hashlib.sha1(str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", input_path.stem).strip("._") or "input"
    return output_root / f"{safe_stem}__{digest}"


def term(name: str) -> str:
    return f"C({name})" if name in CATEGORICAL_COLUMNS else name


def build_fixed_formula(predictors: list[str], pairwise_interactions: bool) -> str:
    terms = [term(name) for name in predictors]
    if not pairwise_interactions:
        return " + ".join(terms)

    parts = list(terms)
    parts.extend(f"{left}:{right}" for left, right in combinations(terms, 2))
    return " + ".join(parts)


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

    for column in REQUIRED_COLUMNS | OPTIONAL_NUMERIC_COLUMNS:
        if column in df.columns and column not in CATEGORICAL_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].astype("category")

    df = df.dropna(subset=sorted(REQUIRED_COLUMNS)).copy()
    return df


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

    return {
        "R2m": var_fixed / total,
        "R2c": (var_fixed + var_random) / total,
    }


def fit_mixed_model(
    df: pd.DataFrame,
    spec: ModelSpec,
    output_dir: Path,
    maxiter: int,
) -> dict[str, Any]:
    model_dir = output_dir / "models" / spec.name
    model_dir.mkdir(parents=True, exist_ok=True)

    formula = (
        "log_he_she_odds ~ "
        f"{build_fixed_formula(spec.fixed_predictors, spec.pairwise_interactions)}"
    )
    re_formula = build_random_formula(spec.random_predictors)

    fit_errors: list[dict[str, str]] = []
    methods = ["lbfgs", "bfgs", "cg", "powell", "nm"]

    for method in methods:
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

            covariance = pd.DataFrame(
                np.asarray(result.cov_re),
                index=result.cov_re.index,
                columns=result.cov_re.columns,
            )
            covariance.index.name = "random_term"

            random_effects = pd.DataFrame.from_dict(result.random_effects, orient="index")
            random_effects.index.name = spec.group_col

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

            summary_text = result.summary().as_text()
            if caught_warnings:
                summary_text += "\n\nWarnings:\n"
                summary_text += "\n".join(
                    f"- {warning.category.__name__}: {warning.message}"
                    for warning in caught_warnings
                )

            fixed_effects.to_csv(model_dir / "fixed_effects.csv")
            covariance.to_csv(model_dir / "random_effects_covariance.csv")
            random_effects.to_csv(model_dir / "group_random_effects.csv")
            with (model_dir / "metrics.json").open("w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)
            with (model_dir / "summary.txt").open("w", encoding="utf-8") as fh:
                fh.write(summary_text)

            return {
                "status": "ok",
                "spec": spec,
                "result": result,
                "metrics": metrics,
                "warnings": [
                    f"{warning.category.__name__}: {warning.message}"
                    for warning in caught_warnings
                ],
            }
        except Exception as exc:  # pragma: no cover - depends on optimiser/runtime
            fit_errors.append({"optimizer": method, "error": repr(exc)})

    failure = {
        "status": "failed",
        "spec": asdict(spec),
        "formula": formula,
        "re_formula": re_formula,
        "fit_errors": fit_errors,
    }
    with (model_dir / "fit_failed.json").open("w", encoding="utf-8") as fh:
        json.dump(failure, fh, indent=2)
    return failure


def likelihood_ratio_test(
    full_result: Any,
    reduced_result: Any,
    full_name: str,
    reduced_name: str,
) -> dict[str, float | int | str]:
    lr_stat = 2.0 * (full_result.llf - reduced_result.llf)
    df_diff = int(full_result.df_modelwc - reduced_result.df_modelwc)
    p_value = float(chi2.sf(lr_stat, df_diff)) if df_diff > 0 else float("nan")
    return {
        "full_model": full_name,
        "reduced_model": reduced_name,
        "lr_stat": float(lr_stat),
        "df_diff": df_diff,
        "p_value": p_value,
    }


def clean_profession_name(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.lower().str.strip()
    return cleaned.str.replace(r"[^A-Za-z0-9]+", "", regex=True)


def load_profession_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    expected = {"profession", "female_percentage"}
    missing = sorted(expected - set(metadata.columns))
    if missing:
        raise ValueError(
            f"Profession metadata CSV is missing required columns: {', '.join(missing)}"
        )
    metadata = metadata.copy()
    metadata["profession"] = clean_profession_name(metadata["profession"])
    metadata["female_percentage"] = pd.to_numeric(
        metadata["female_percentage"], errors="coerce"
    )
    metadata = metadata.dropna(subset=["profession", "female_percentage"])
    metadata = metadata.drop_duplicates(subset=["profession"])
    return metadata


def merge_profession_metadata(df: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    combined = df.copy()
    combined["profession_original"] = combined["profession"].astype("string")
    combined["profession"] = clean_profession_name(combined["profession"])
    combined = combined.merge(metadata, on="profession", how="left")

    unmatched = sorted(
        combined.loc[combined["female_percentage"].isna(), "profession"].dropna().unique().tolist()
    )
    return combined, unmatched


def predict_with_random_intercept(result: Any, df: pd.DataFrame, group_col: str) -> np.ndarray:
    predictions = np.asarray(result.predict(df), dtype=float)
    random_effects = getattr(result, "random_effects", {})
    if not random_effects:
        return predictions

    intercepts: dict[str, float] = {}
    for group, values in random_effects.items():
        if hasattr(values, "iloc"):
            intercept_value = float(values.iloc[0])
        else:
            intercept_value = float(np.asarray(values)[0])
        intercepts[str(group)] = intercept_value

    group_offsets = df[group_col].astype("string").map(intercepts).fillna(0.0).to_numpy(dtype=float)
    return predictions + group_offsets


def cross_validate_random_intercept_model(
    df: pd.DataFrame,
    fixed_predictors: list[str],
    group_col: str,
    output_path: Path,
    maxiter: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    formula = "log_he_she_odds ~ " + build_fixed_formula(
        fixed_predictors, pairwise_interactions=False
    )

    for fold, (train_idx, test_idx) in enumerate(
        progress_iterable(splitter.split(df), total=5, desc=f"CV {output_path.stem}", unit="fold"),
        start=1,
    ):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        model = smf.mixedlm(formula=formula, data=train_df, groups=train_df[group_col], re_formula="1")
        result = model.fit(reml=False, method="lbfgs", maxiter=maxiter, disp=False)
        predictions = predict_with_random_intercept(result, test_df, group_col=group_col)
        observed = test_df["log_he_she_odds"].to_numpy(dtype=float)

        rows.append(
            {
                "fold": fold,
                "rmse": float(np.sqrt(mean_squared_error(observed, predictions))),
                "mae": float(mean_absolute_error(observed, predictions)),
                "r2": float(r2_score(observed, predictions)),
            }
        )

    fold_metrics = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "fold": "mean",
                "rmse": float(fold_metrics["rmse"].mean()),
                "mae": float(fold_metrics["mae"].mean()),
                "r2": float(fold_metrics["r2"].mean()),
            }
        ]
    )
    combined = pd.concat([fold_metrics, summary], ignore_index=True)
    combined.to_csv(output_path, index=False)
    return combined


def run_primary_analysis(df: pd.DataFrame, output_dir: Path, maxiter: int) -> dict[str, Any]:
    specs = [
        ModelSpec(
            name="full_mixed",
            fixed_predictors=["tense", "syntactic_role", "valence", "dominance", "semantic_role"],
            group_col="profession",
            random_predictors=["semantic_role", "syntactic_role"],
        ),
        ModelSpec(
            name="no_tense",
            fixed_predictors=["syntactic_role", "valence", "dominance", "semantic_role"],
            group_col="profession",
            random_predictors=["semantic_role", "syntactic_role"],
        ),
        ModelSpec(
            name="no_syntactic_role",
            fixed_predictors=["tense", "valence", "dominance", "semantic_role"],
            group_col="profession",
            random_predictors=["semantic_role"],
        ),
        ModelSpec(
            name="no_valence",
            fixed_predictors=["tense", "syntactic_role", "dominance", "semantic_role"],
            group_col="profession",
            random_predictors=["semantic_role", "syntactic_role"],
        ),
        ModelSpec(
            name="no_dominance",
            fixed_predictors=["tense", "syntactic_role", "valence", "semantic_role"],
            group_col="profession",
            random_predictors=["semantic_role", "syntactic_role"],
        ),
        ModelSpec(
            name="no_semantic_role",
            fixed_predictors=["tense", "syntactic_role", "valence", "dominance"],
            group_col="profession",
            random_predictors=["syntactic_role"],
        ),
        ModelSpec(
            name="full_mixed_2",
            fixed_predictors=["tense", "syntactic_role", "valence", "dominance", "profession"],
            group_col="semantic_role",
            random_predictors=["profession", "syntactic_role"],
        ),
        ModelSpec(
            name="no_profession",
            fixed_predictors=["tense", "valence", "dominance", "syntactic_role"],
            group_col="semantic_role",
            random_predictors=["syntactic_role"],
        ),
    ]

    fits: dict[str, Any] = {}
    for spec in progress_iterable(specs, total=len(specs), desc="Primary models", unit="model"):
        fits[spec.name] = fit_mixed_model(df, spec, output_dir, maxiter=maxiter)

    comparisons: list[dict[str, float | int | str]] = []
    full = fits["full_mixed"]
    if full["status"] == "ok":
        for reduced_name in [
            "no_tense",
            "no_syntactic_role",
            "no_valence",
            "no_dominance",
            "no_semantic_role",
        ]:
            reduced = fits[reduced_name]
            if reduced["status"] == "ok":
                comparisons.append(
                    likelihood_ratio_test(
                        full["result"],
                        reduced["result"],
                        full_name="full_mixed",
                        reduced_name=reduced_name,
                    )
                )

    full_2 = fits["full_mixed_2"]
    no_profession = fits["no_profession"]
    if full_2["status"] == "ok" and no_profession["status"] == "ok":
        comparisons.append(
            likelihood_ratio_test(
                full_2["result"],
                no_profession["result"],
                full_name="full_mixed_2",
                reduced_name="no_profession",
            )
        )

    if comparisons:
        pd.DataFrame(comparisons).to_csv(output_dir / "model_comparisons.csv", index=False)

    return {
        "models": {
            name: {
                "status": fit["status"],
                "group_col": fit["spec"].group_col,
            }
            for name, fit in fits.items()
        },
        "comparisons_written": bool(comparisons),
    }


def run_metadata_analysis(
    df: pd.DataFrame,
    metadata_csv: Path,
    output_dir: Path,
    maxiter: int,
) -> dict[str, Any]:
    metadata_dir = output_dir / "metadata_models"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_profession_metadata(metadata_csv)
    combined, unmatched = merge_profession_metadata(df, metadata)
    combined.to_csv(metadata_dir / "combined_results_with_metadata.csv", index=False)

    with (metadata_dir / "merge_report.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "metadata_csv": str(metadata_csv.resolve()),
                "matched_rows": int(combined["female_percentage"].notna().sum()),
                "unmatched_professions": unmatched,
            },
            fh,
            indent=2,
        )

    analysis_df = combined.dropna(subset=["female_percentage"]).copy()
    analysis_df["female_percentage"] = pd.to_numeric(
        analysis_df["female_percentage"], errors="coerce"
    )
    analysis_df = analysis_df.dropna(subset=["female_percentage"]).copy()

    if analysis_df.empty:
        return {
            "status": "skipped",
            "reason": "No rows remained after merging female_percentage metadata.",
            "unmatched_professions": unmatched,
        }

    specs = [
        ModelSpec(
            name="full_mixed_3",
            fixed_predictors=[
                "tense",
                "syntactic_role",
                "valence",
                "dominance",
                "female_percentage",
            ],
            group_col="semantic_role",
            random_predictors=[],
            pairwise_interactions=False,
        ),
        ModelSpec(
            name="no_female_percentage",
            fixed_predictors=["tense", "syntactic_role", "valence", "dominance"],
            group_col="semantic_role",
            random_predictors=[],
            pairwise_interactions=False,
        ),
        ModelSpec(
            name="full_mixed_4",
            fixed_predictors=["tense", "syntactic_role", "valence", "dominance", "profession"],
            group_col="semantic_role",
            random_predictors=[],
            pairwise_interactions=False,
        ),
    ]

    fits: dict[str, Any] = {}
    for spec in progress_iterable(specs, total=len(specs), desc="Metadata models", unit="model"):
        fits[spec.name] = fit_mixed_model(analysis_df, spec, metadata_dir, maxiter=maxiter)

    comparisons: list[dict[str, float | int | str]] = []
    full_3 = fits["full_mixed_3"]
    no_female = fits["no_female_percentage"]
    full_4 = fits["full_mixed_4"]

    if full_3["status"] == "ok" and no_female["status"] == "ok":
        comparisons.append(
            likelihood_ratio_test(
                full_3["result"],
                no_female["result"],
                full_name="full_mixed_3",
                reduced_name="no_female_percentage",
            )
        )

    if full_3["status"] == "ok" and full_4["status"] == "ok":
        pd.DataFrame(
            [
                {
                    "model": "full_mixed_3",
                    "aic": full_3["metrics"]["aic"],
                    "bic": full_3["metrics"]["bic"],
                },
                {
                    "model": "full_mixed_4",
                    "aic": full_4["metrics"]["aic"],
                    "bic": full_4["metrics"]["bic"],
                },
            ]
        ).to_csv(metadata_dir / "aic_bic_comparison.csv", index=False)

        cv_failures: list[dict[str, str]] = []
        cv_specs = [
            (
                "female_percentage",
                ["tense", "syntactic_role", "valence", "dominance", "female_percentage"],
                metadata_dir / "cross_validation_female_percentage.csv",
            ),
            (
                "profession",
                ["tense", "syntactic_role", "valence", "dominance", "profession"],
                metadata_dir / "cross_validation_profession.csv",
            ),
        ]
        for label, fixed_predictors, output_path in progress_iterable(
            cv_specs, total=len(cv_specs), desc="Cross-validation", unit="run"
        ):
            try:
                cross_validate_random_intercept_model(
                    analysis_df,
                    fixed_predictors=fixed_predictors,
                    group_col="semantic_role",
                    output_path=output_path,
                    maxiter=maxiter,
                )
            except Exception as exc:  # pragma: no cover - runtime dependent
                cv_failures.append({"model": label, "error": repr(exc)})

        if cv_failures:
            with (metadata_dir / "cross_validation_failed.json").open(
                "w", encoding="utf-8"
            ) as fh:
                json.dump(cv_failures, fh, indent=2)

    if comparisons:
        pd.DataFrame(comparisons).to_csv(metadata_dir / "model_comparisons.csv", index=False)

    return {
        "rows_with_metadata": int(len(analysis_df)),
        "unmatched_professions": unmatched,
        "models": {name: {"status": fit["status"]} for name, fit in fits.items()},
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

    data_profile = {
        "input_csv": str(args.results_csv.resolve()),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "category_levels": {
            column: df[column].astype("string").dropna().sort_values().unique().tolist()
            for column in sorted(CATEGORICAL_COLUMNS)
            if column in df.columns
        },
    }
    with (output_dir / "data_profile.json").open("w", encoding="utf-8") as fh:
        json.dump(data_profile, fh, indent=2)

    run_summary: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_analysis": run_primary_analysis(df, output_dir, maxiter=args.maxiter),
    }

    if args.profession_metadata_csv is not None:
        if not args.profession_metadata_csv.exists():
            print(
                f"Profession metadata CSV not found: {args.profession_metadata_csv}",
                file=sys.stderr,
            )
            return 1
        run_summary["metadata_analysis"] = run_metadata_analysis(
            df,
            args.profession_metadata_csv,
            output_dir,
            maxiter=args.maxiter,
        )
    else:
        run_summary["metadata_analysis"] = {
            "status": "skipped",
            "reason": "No --profession-metadata-csv argument supplied.",
        }

    run_summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(run_summary, fh, indent=2)

    print(f"Analysis complete. Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
