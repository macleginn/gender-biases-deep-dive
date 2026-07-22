#!/usr/bin/env python3
"""Compare model fits produced by dissertation_analysing_results.py.

The script scans analysis output directories, reads each model's metrics.json,
and writes a consolidated comparison table plus simple rankings.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare model fit metrics across dissertation_analysing_results.py outputs."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("analysis_outputs"),
        help=(
            "Analysis output root or a single run directory. "
            "Default: analysis_outputs"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/model_comparisons"),
        help="Directory for the comparison tables. Default: analysis_outputs/model_comparisons",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_target_model(run_name: str, input_csv: str | None) -> str:
    candidates: list[str] = []
    if input_csv:
        candidates.append(Path(input_csv).stem)
    candidates.append(run_name)

    for candidate in candidates:
        parts = candidate.split("__")
        if len(parts) >= 3:
            return parts[1]
    return run_name


def iter_run_dirs(root: Path) -> list[Path]:
    if (root / "run_summary.json").exists():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    return sorted(
        candidate
        for candidate in root.iterdir()
        if candidate.is_dir() and (candidate / "run_summary.json").exists()
    )


def collect_model_rows(run_dir: Path) -> list[dict[str, Any]]:
    run_summary = load_json(run_dir / "run_summary.json")
    input_csv = run_summary.get("input_csv")
    rows: list[dict[str, Any]] = []

    models_dir = run_dir / "models"
    if not models_dir.exists():
        return rows

    for model_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        fixed_effects_path = model_dir / "fixed_effects.csv"
        rows.append(
            {
                "run_dir": str(run_dir.resolve()),
                "run_name": run_dir.name,
                "input_csv": input_csv,
                "target_model": extract_target_model(run_dir.name, input_csv),
                "model_name": model_dir.name,
                "model_dir": str(model_dir.resolve()),
                "fixed_effects_path": str(fixed_effects_path.resolve()) if fixed_effects_path.exists() else None,
                **metrics,
            }
        )

    metadata_dir = run_dir / "metadata_models"
    if metadata_dir.exists():
        for model_dir in sorted(p for p in metadata_dir.iterdir() if p.is_dir()):
            metrics_path = model_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = load_json(metrics_path)
            fixed_effects_path = model_dir / "fixed_effects.csv"
            rows.append(
                {
                    "run_dir": str(run_dir.resolve()),
                    "run_name": run_dir.name,
                    "input_csv": input_csv,
                    "target_model": extract_target_model(run_dir.name, input_csv),
                    "model_name": model_dir.name,
                    "model_dir": str(model_dir.resolve()),
                    "fixed_effects_path": str(fixed_effects_path.resolve()) if fixed_effects_path.exists() else None,
                    **metrics,
                }
            )

    return rows


def rank_models(df: pd.DataFrame, metric: str, ascending: bool) -> pd.DataFrame:
    ranked = df.copy()
    ranked[f"{metric}_rank"] = ranked.groupby(["target_model", "run_name"])[metric].rank(
        method="min", ascending=ascending
    )
    return ranked


def load_fixed_effects(model_dir: str) -> pd.DataFrame | None:
    path = Path(model_dir) / "fixed_effects.csv"
    if not path.exists():
        return None
    fixed = pd.read_csv(path)
    if "term" not in fixed.columns or "coef" not in fixed.columns:
        return None
    return fixed


def build_coefficient_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, model_row in df.iterrows():
        fixed = load_fixed_effects(model_row["model_dir"])
        if fixed is None:
            continue
        fixed = fixed.copy()
        fixed["coef"] = pd.to_numeric(fixed["coef"], errors="coerce")
        for _, coef_row in fixed.iterrows():
            rows.append(
                {
                    "run_name": model_row["run_name"],
                    "target_model": model_row["target_model"],
                    "model_name": model_row["model_name"],
                    "term": coef_row["term"],
                    "coef": coef_row["coef"],
                    "std_err": pd.to_numeric(coef_row.get("std_err"), errors="coerce"),
                    "p_value": pd.to_numeric(coef_row.get("p_value"), errors="coerce"),
                    "ci_low": pd.to_numeric(coef_row.get("ci_low"), errors="coerce"),
                    "ci_high": pd.to_numeric(coef_row.get("ci_high"), errors="coerce"),
                }
            )
    return pd.DataFrame(rows)


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def plot_fit_overview(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return

    plot_df = summary.dropna(subset=["mean_aic", "mean_R2c"]).copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        plot_df["mean_aic"],
        plot_df["mean_R2c"],
        c=plot_df["target_model"].astype("category").cat.codes,
        cmap="tab10",
        s=80,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
    )

    for _, row in plot_df.iterrows():
        ax.annotate(
            row["model_name"],
            (row["mean_aic"], row["mean_R2c"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Mean AIC")
    ax.set_ylabel("Mean conditional R²")
    ax.set_title("Model fit overview across target models")
    ax.grid(True, alpha=0.25)

    handles, labels = scatter.legend_elements()
    target_labels = list(plot_df["target_model"].astype("category").cat.categories)
    if len(handles) == len(target_labels):
        ax.legend(handles, target_labels, title="Target model", loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / "model_fit_overview.png", dpi=200)
    plt.close(fig)


def plot_summary_across_models(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    saved: list[Path] = []
    if summary.empty:
        return saved

    plot_metrics: list[tuple[str, str, str, bool]] = [
        ("mean_aic", "Mean AIC", "lower is better", True),
        ("mean_bic", "Mean BIC", "lower is better", True),
        ("mean_R2c", "Mean conditional R²", "higher is better", False),
        ("converged_rate", "Convergence rate", "higher is better", False),
    ]

    available = [(col, title, subtitle, invert) for col, title, subtitle, invert in plot_metrics if col in summary.columns]
    if not available:
        return saved

    target_models = list(summary["target_model"].dropna().astype(str).sort_values().unique())
    if not target_models:
        return saved

    model_names = list(summary["model_name"].dropna().astype(str).sort_values().unique())
    if not model_names:
        return saved

    width = max(10.0, 1.25 * len(target_models) * max(1, len(model_names) / 2))
    palette = plt.get_cmap("tab20")(range(max(1, len(model_names))))
    model_colors = {model_name: palette[i % len(palette)] for i, model_name in enumerate(model_names)}

    for column, title, subtitle, invert in available:
        plot_df = summary.dropna(subset=[column]).copy()
        if plot_df.empty:
            continue

        pivot = plot_df.pivot_table(
            index="target_model",
            columns="model_name",
            values=column,
            aggfunc="mean",
        ).reindex(index=target_models, columns=model_names)
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(width, 6))
        x = range(len(pivot.index))
        n_series = len(pivot.columns)
        bar_width = 0.8 / max(1, n_series)
        offsets = [(-0.4 + bar_width / 2) + i * bar_width for i in range(n_series)]

        for i, model_name in enumerate(pivot.columns):
            ax.bar(
                [pos + offsets[i] for pos in x],
                pivot[model_name].to_numpy(),
                width=bar_width,
                label=model_name,
                color=model_colors.get(model_name),
                edgecolor="black",
                linewidth=0.4,
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels(pivot.index, rotation=30, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"{title} across target models")
        ax.grid(axis="y", alpha=0.25)
        if invert:
            ax.invert_yaxis()
        if subtitle:
            ax.text(
                0.99,
                0.94,
                subtitle,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="dimgray",
            )
        ax.legend(title="Model type", ncols=min(3, len(model_names)), fontsize=8)
        fig.tight_layout()

        out_path = output_dir / f"summary_across_target_models__{safe_filename(column)}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        saved.append(out_path)

    return saved


def plot_coefficient_heatmaps(coef_summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    saved: list[Path] = []
    if coef_summary.empty:
        return saved

    for target_model, target_df in coef_summary.groupby("target_model", dropna=False):
        pivot = target_df.pivot_table(
            index="term",
            columns="model_name",
            values="mean_coef",
            aggfunc="mean",
        )
        if pivot.empty:
            continue

        pivot = pivot.sort_index()
        fig_width = max(8, 0.7 * len(pivot.columns) + 4)
        fig_height = max(6, 0.35 * len(pivot.index) + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", interpolation="nearest")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Mean coefficient by model type for {target_model}")
        ax.set_xlabel("Model type")
        ax.set_ylabel("Coefficient term")
        fig.colorbar(im, ax=ax, label="Mean coefficient")
        fig.tight_layout()

        out_path = output_dir / f"coefficient_heatmap__{safe_filename(str(target_model))}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        saved.append(out_path)

    return saved


def main() -> int:
    args = parse_args()

    try:
        run_dirs = iter_run_dirs(args.input)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.extend(collect_model_rows(run_dir))

    if not rows:
        print(f"No model metrics found under: {args.input}", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    for column in ["aic", "bic", "log_likelihood", "residual_variance", "R2m", "R2c", "nobs"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "converged" in df.columns:
        df["converged"] = df["converged"].astype("boolean")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_csv = output_dir / "model_metrics_across_runs.csv"
    df.sort_values(["model_name", "run_name"]).to_csv(comparison_csv, index=False)

    ranking_rows = []
    if "aic" in df.columns:
        ranking_rows.append(rank_models(df, "aic", ascending=True))
    if "bic" in df.columns:
        ranking_rows.append(rank_models(df, "bic", ascending=True))
    if "R2c" in df.columns:
        ranking_rows.append(rank_models(df, "R2c", ascending=False))

    if ranking_rows:
        rankings = pd.concat(ranking_rows, ignore_index=True)
        rankings.to_csv(output_dir / "model_rankings.csv", index=False)

    summary = (
        df.groupby(["target_model", "model_name"], dropna=False)
        .agg(
            runs=("run_name", "nunique"),
            mean_aic=("aic", "mean") if "aic" in df.columns else ("model_name", "size"),
            mean_bic=("bic", "mean") if "bic" in df.columns else ("model_name", "size"),
            mean_R2c=("R2c", "mean") if "R2c" in df.columns else ("model_name", "size"),
            converged_rate=("converged", "mean") if "converged" in df.columns else ("model_name", "size"),
        )
        .reset_index()
        .sort_values(["target_model", "mean_aic", "mean_bic"], na_position="last")
    )
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    plot_fit_overview(summary, output_dir)
    summary_plots = plot_summary_across_models(summary, output_dir)

    coef_df = build_coefficient_rows(df)
    if not coef_df.empty:
        coef_df = coef_df.dropna(subset=["coef"])
        coef_summary = (
            coef_df.groupby(["target_model", "model_name", "term"], dropna=False)
            .agg(
                n_runs=("run_name", "nunique"),
                mean_coef=("coef", "mean"),
                std_coef=("coef", "std"),
                min_coef=("coef", "min"),
                max_coef=("coef", "max"),
                positive_rate=("coef", lambda s: (s > 0).mean()),
                negative_rate=("coef", lambda s: (s < 0).mean()),
            )
            .reset_index()
            .sort_values(["target_model", "model_name", "term"])
        )
        coef_summary.to_csv(output_dir / "coefficient_stability_by_target_model.csv", index=False)

        cross_model_summary = (
            coef_df.groupby(["model_name", "term"], dropna=False)
            .agg(
                target_models=("target_model", "nunique"),
                n_runs=("run_name", "nunique"),
                mean_coef=("coef", "mean"),
                std_coef=("coef", "std"),
                min_coef=("coef", "min"),
                max_coef=("coef", "max"),
                sign_consistency=("coef", lambda s: max((s > 0).mean(), (s < 0).mean())),
            )
            .reset_index()
            .sort_values(["model_name", "term"])
        )
        cross_model_summary.to_csv(output_dir / "coefficient_stability_across_model_types.csv", index=False)
        heatmaps = plot_coefficient_heatmaps(coef_summary, output_dir)

    print(f"Wrote comparison tables to: {output_dir}")
    print(f"- {comparison_csv.name}")
    if ranking_rows:
        print("- model_rankings.csv")
    print("- model_summary.csv")
    print("- model_fit_overview.png")
    for path in summary_plots:
        print(f"- {path.name}")
    if not coef_df.empty:
        print("- coefficient_stability_by_target_model.csv")
        print("- coefficient_stability_across_model_types.csv")
        for path in heatmaps:
            print(f"- {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
