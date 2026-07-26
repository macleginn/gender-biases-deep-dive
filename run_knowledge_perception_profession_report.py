#!/usr/bin/env python3
"""Fit and report the World-Knowledge and Human-Perception profession models.

This report follows the LassoCV-selection/OLS-refit workflow in
``run_lasso_profession_embeddings_report.py`` but uses profession male-
membership frequency from ``professions.csv`` instead of profession
embeddings.  The two candidate models are::

    log_he_she_odds ~ (MalePerc + semantic_role + valence + dominance)^2
    log_he_she_odds ~ (MalePerc + syntactic_role + valence + dominance)^2

Categorical predictors are expanded with treatment coding by patsy, and all
pairwise interactions are available to LassoCV.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import textwrap
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any

import run_lasso_profession_embeddings_report as base


if base.IMPORT_ERROR is not None:  # pragma: no cover
    IMPORT_ERROR = base.IMPORT_ERROR
else:
    IMPORT_ERROR = None


MODEL_SPECS = {
    "world_knowledge": {
        "label": "World-Knowledge Model",
        "role": "semantic_role",
    },
    "human_perception": {
        "label": "Human-Perception Model",
        "role": "syntactic_role",
    },
}
PREDICTORS = ("male_perc", "valence", "dominance")
NUMERICAL_PREDICTORS = ("male_perc",)
PREPROCESSING_VERSION = "zscore_numeric-model-predictors_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path, nargs="*")
    parser.add_argument("--professions", type=Path, default=Path("professions.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("knowledge_perception_model_data"))
    parser.add_argument("--report-dir", type=Path, default=Path("knowledge_perception_model_report"))
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def model_order_with_pair_gaps(model_order: list[str]) -> list[str]:
    """Insert a blank plotting row between each pair of model variants."""
    order: list[str] = []
    for index, model in enumerate(model_order):
        order.append(model)
        if index % 2 == 1 and index < len(model_order) - 1:
            order.append(f"__pair_gap_{index // 2}__")
    return order


def format_paired_bar_axis(ax: Any, plot_order: list[str]) -> None:
    """Hide gap-row labels and mark the widened spaces between model pairs."""
    ax.set_yticks(
        range(len(plot_order)),
        labels=["" if label.startswith("__pair_gap_") else label for label in plot_order],
    )
    for position, label in enumerate(plot_order):
        if label.startswith("__pair_gap_"):
            ax.axhline(position, color="#b8b1a5", linewidth=1.2, zorder=0)


def load_profession_metadata(path: Path) -> tuple[base.pd.DataFrame, dict[str, Any]]:
    metadata = base.pd.read_csv(path)
    if metadata.shape[1] < 2:
        raise ValueError(f"{path} must contain a profession column and MalePerc")
    profession_column = metadata.columns[0]
    if "MalePerc" not in metadata.columns:
        raise ValueError(f"{path} must contain a MalePerc column")
    names = metadata[profession_column].astype("string").str.strip()
    if names.isna().any() or names.duplicated().any():
        raise ValueError(f"{path} contains missing or duplicate profession names")
    male_perc = base.pd.to_numeric(metadata["MalePerc"], errors="coerce")
    if male_perc.isna().any():
        raise ValueError(f"{path} contains non-numeric MalePerc values")
    if ((male_perc < 0) | (male_perc > 100)).any():
        raise ValueError(f"{path} contains MalePerc values outside 0-100")
    result = base.pd.DataFrame({"profession": names, "male_perc": male_perc})
    return result, {
        "source": str(path.resolve()),
        "profession_column": str(profession_column),
        "rows": int(len(result)),
        "predictor": "MalePerc",
    }


def prepare_data(path: Path, metadata: base.pd.DataFrame) -> base.pd.DataFrame:
    data = base.load_results_csv(path)
    data["profession"] = data["profession"].astype("string").str.strip()
    known = set(metadata["profession"].astype(str))
    unknown = sorted(set(data["profession"].dropna().astype(str)) - known)
    if unknown:
        raise ValueError(f"{path} contains professions absent from professions.csv: {unknown[:10]}")
    merged = data.merge(metadata, on="profession", how="inner", validate="many_to_one")
    scaling = base.standardize_predictors(merged, list(NUMERICAL_PREDICTORS), source=path)
    merged.attrs["preprocessing"] = {
        "version": PREPROCESSING_VERSION,
        "standardized_predictors": scaling,
        "note": "log_frequency and lex_emb_norm are not model predictors in these specifications.",
    }
    return merged


def model_terms(role: str) -> list[str]:
    names = ["male_perc", role, "valence", "dominance"]
    main = [base.term(name) for name in names]
    return main + [f"{a}:{b}" for a, b in combinations(main, 2)]


def select_model(df: base.pd.DataFrame, run_dir: Path, role: str, maxiter: int) -> tuple[dict[str, Any], list[str], base.pd.DataFrame]:
    run_dir.mkdir(parents=True, exist_ok=True)
    all_terms = model_terms(role)
    formula = "log_he_she_odds ~ " + " + ".join(all_terms)
    response, design = base.dmatrices(formula, data=df, return_type="dataframe")
    x_scaled = base.StandardScaler().fit_transform(design)
    y = base.np.asarray(response).ravel()
    lasso = base.LassoCV(cv=5, max_iter=maxiter, n_jobs=-1, random_state=0).fit(x_scaled, y)
    base.pd.DataFrame({
        "feature": list(design.columns),
        "coefficient": lasso.coef_,
        "nonzero": base.np.abs(lasso.coef_) > 1e-10,
    }).to_csv(run_dir / "lasso_coefficients.csv", index=False)

    selected_terms = []
    for term_name, term_slice in design.design_info.term_name_slices.items():
        if term_name != "Intercept" and (base.np.abs(lasso.coef_[term_slice]) > 1e-10).any():
            selected_terms.append(term_name)
    selected = base.fit_model(df, "selected_model", selected_terms, [], run_dir)
    decomposition = base.decompose_r_squared(df, selected, selected_terms, run_dir)
    (run_dir / "selected_model.json").write_text(json.dumps({
        "selected_model": selected["name"],
        "candidate_formula": formula,
        "selected_terms": selected_terms,
        "alpha": float(lasso.alpha_),
        "alpha_selection": "LassoCV",
        "cv": 5,
        "lasso_n_iter": int(lasso.n_iter_),
    }, indent=2), encoding="utf-8")
    return selected, selected_terms, decomposition


def run_one(input_csv: Path, metadata: base.pd.DataFrame, metadata_meta: dict[str, Any],
            data_dir: Path, maxiter: int) -> Path:
    run_dir = data_dir / "runs" / slugify(input_csv.stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "inputs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_csv, data_dir / "inputs" / input_csv.name)
    df = prepare_data(input_csv, metadata)
    preprocessing = df.attrs.get("preprocessing", {})
    df.to_csv(run_dir / "prepared_results.csv", index=False)
    selected_models = {}
    for key, spec in MODEL_SPECS.items():
        selected, terms, decomposition = select_model(df, run_dir / key, spec["role"], maxiter)
        selected_models[key] = {
            "label": spec["label"], "role": spec["role"], "model_dir": str((run_dir / key / "models" / "selected_model").resolve()),
            "run_dir": str((run_dir / key).resolve()), "selected_terms": terms,
            "selected_predictor_count": int(len(decomposition)),
            "name": selected["name"],
        }
    (run_dir / "run_summary.json").write_text(json.dumps({
        "input_csv": str(input_csv.resolve()), "profession_metadata": metadata_meta,
        "models": selected_models, "preprocessing": preprocessing,
    }, indent=2), encoding="utf-8")
    return run_dir


def discover_inputs(paths: list[Path]) -> list[Path]:
    inputs = paths or sorted(Path("modelling_data").glob("he_she_odds_results__*.csv"))
    if not inputs:
        raise FileNotFoundError("No input CSVs found")
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input CSVs: " + ", ".join(missing))
    return [path.resolve() for path in inputs]


def build_report(run_dirs: list[Path], report_dir: Path, metadata_meta: dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    table_dir = report_dir / "tables"
    figure_dir = report_dir / "figures"
    table_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    coefficient_frames = []
    decomposition_frames = []
    for run in run_dirs:
        summary = json.loads((run / "run_summary.json").read_text())
        target = base.clean_model_name(Path(summary["input_csv"]).stem)
        for key, model in summary["models"].items():
            model_dir = Path(model["model_dir"])
            metrics = json.loads((model_dir / "metrics.json").read_text())
            rows.append({"target_model": target, "model": model["label"], "model_key": key, "model_dir": str(model_dir), **metrics})
            fixed = base.pd.read_csv(model_dir / "fixed_effects.csv")
            fixed.insert(0, "model", model["label"])
            fixed.insert(0, "target_model", target)
            coefficient_frames.append(fixed)
            decomposition = base.pd.read_csv(Path(model["run_dir"]) / "r2_decomposition.csv")
            decomposition.insert(0, "model", model["label"])
            decomposition.insert(0, "target_model", target)
            decomposition_frames.append(decomposition)
    metrics = base.pd.DataFrame(rows).sort_values(["target_model", "model"])
    coefficients = base.pd.concat(coefficient_frames, ignore_index=True)
    decomposition = base.pd.concat(decomposition_frames, ignore_index=True)
    metrics.to_csv(report_dir / "model_metrics.csv", index=False)
    decomposition.to_csv(report_dir / "r2_decomposition.csv", index=False)
    available_inputs = set(metrics["target_model"])
    configured_order, configured_labels = base.load_model_display_config()
    input_order = [name for name in configured_order if name in available_inputs]
    input_order.extend(sorted(available_inputs - set(input_order)))
    input_labels = {name: configured_labels.get(name, name) for name in input_order}
    plot_input_order = [input_labels[name] for name in input_order]
    variance_cols = [column for column in metrics.columns if column.startswith("fixed_effect_variance_")]
    variance = metrics[["target_model", "model", *variance_cols]].copy()
    variance = variance.rename(columns={
        "target_model": "Input",
        "model": "Model",
        **{
            column: column.removeprefix("fixed_effect_variance_").replace("_", " ")
            for column in variance_cols
        },
    })
    variance["Input"] = variance["Input"].map(input_labels)
    variance.columns = [
        column if column in {"Input", "Model"} else base.fixed_effect_label(column)
        for column in variance.columns
    ]
    coefficients["term"] = coefficients["term"].map(base.fixed_effect_label)
    coefficients.to_csv(report_dir / "selected_coefficients.csv", index=False)
    variance.to_csv(report_dir / "variance_decomposition.csv", index=False)
    fit = metrics[["target_model", "model", "nobs", "aic", "bic", "R2m", "residual_variance"]].rename(columns={
        "target_model": "Input", "model": "Model", "nobs": "N", "R2m": "R2", "residual_variance": "Residual variance",
    })
    fit["Input"] = fit["Input"].map(input_labels)
    fit["Plot model"] = fit["Input"] + ": " + fit["Model"]
    plot_model_order = [
        f"{input_label}: {model_label}"
        for input_label in plot_input_order
        for model_label in [spec["label"] for spec in MODEL_SPECS.values()]
        if f"{input_label}: {model_label}" in set(fit["Plot model"])
    ]
    plot_model_order_with_gaps = model_order_with_pair_gaps(plot_model_order)
    plot_model_order_reversed_with_gaps = plot_model_order_with_gaps[::-1]
    fit_html = base.write_table(fit.drop(columns=["Plot model"]), table_dir / "model_fit.html")
    variance_html = base.write_table(variance, table_dir / "variance_decomposition.html")
    coefficient_table = coefficients.rename(columns={
        "model": "Model", "term": "Term", "coef": "Estimate",
        "std_err": "Std. error", "p_value": "p value", "ci_low": "CI low", "ci_high": "CI high",
    })
    coefficient_html = base.write_table(coefficient_table, table_dir / "selected_coefficients.html", classes="data-table compact")
    decomposition_table = decomposition.copy()
    decomposition_table["target_model"] = decomposition_table["target_model"].map(input_labels)
    decomposition_table["predictor"] = decomposition_table["predictor"].map(base.fixed_effect_label)
    decomposition_html = base.write_table(decomposition_table, table_dir / "r2_decomposition.html", classes="data-table compact")
    base.sns.set_theme(style="whitegrid", context="notebook")
    base.plt.figure(figsize=(8, max(3.5, len(fit) * 0.4 + 1.5)))
    ax = base.sns.barplot(
        data=fit, x="R2", y="Plot model", order=plot_model_order_with_gaps, color="#315c70", width=0.92
    )
    format_paired_bar_axis(ax, plot_model_order_with_gaps)
    base.plt.xlim(0, max(1.0, float(fit["R2"].max()) * 1.08))
    base.plt.xlabel("Explained variance (R2)")
    base.plt.ylabel("")
    base.savefig(figure_dir / "r2_comparison.png")

    r2_by_input = metrics.pivot_table(
        index="target_model", columns="model_key", values="R2m", aggfunc="first"
    ).reindex(input_order)
    human_minus_world = (
        r2_by_input["human_perception"] - r2_by_input["world_knowledge"]
    ).dropna()
    world_r2 = r2_by_input.loc[human_minus_world.index, "world_knowledge"]
    delta_labels = [input_labels[name] for name in human_minus_world.index]
    delta_colors = ["#315c70" if value >= 0 else "#b75d42" for value in human_minus_world]
    fig, ax = base.plt.subplots(figsize=(max(10, len(human_minus_world) * 0.8), 6))
    ax.bar(delta_labels, world_r2.to_numpy(), color="#c8d1d0", label="World-Knowledge R²")
    ax.bar(
        delta_labels,
        human_minus_world.to_numpy(),
        bottom=world_r2.to_numpy(),
        color=delta_colors,
    )
    ax.axhline(0, color="#1d2625", linewidth=1)
    ax.set_xlabel("Target language model")
    ax.set_ylabel("R² (World-Knowledge baseline + Human-Perception change)")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    for position, (baseline, delta) in enumerate(zip(world_r2, human_minus_world)):
        endpoint = baseline + delta
        ax.annotate(
            f"{delta:+.3f}",
            (position, endpoint),
            xytext=(0, 4 if delta >= 0 else -4),
            textcoords="offset points",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            color="#315c70" if delta >= 0 else "#b75d42",
            fontsize=8,
            fontweight="bold",
        )
    ax.legend(
        handles=[
            base.plt.Rectangle((0, 0), 1, 1, color="#c8d1d0", label="World-Knowledge R²"),
            base.plt.Rectangle((0, 0), 1, 1, color="#315c70", label="Positive ΔR²"),
            base.plt.Rectangle((0, 0), 1, 1, color="#b75d42", label="Negative ΔR²"),
        ],
        loc="lower right",
    )
    base.savefig(figure_dir / "r2_delta_human_minus_world.png")
    variance_plot = variance.copy()
    variance_plot["Model"] = variance_plot["Input"] + ": " + variance_plot["Model"]
    variance_plot = variance_plot.set_index("Model").drop(columns="Input")
    variance_plot = variance_plot.div(variance_plot.sum(axis=1).replace(0, base.np.nan), axis=0)
    fixed_effect_names = variance_plot.columns.tolist()
    fixed_effect_numbers = list(range(1, len(fixed_effect_names) + 1))
    fixed_effect_colors = base.sns.color_palette("husl", n_colors=len(fixed_effect_names))
    fig, ax = base.plt.subplots(figsize=(13, max(5.5, len(variance_plot) * 0.9 + 2.5)))
    variance_plot.reindex(plot_model_order_reversed_with_gaps).plot(
        kind="barh", stacked=True, ax=ax, color=fixed_effect_colors, width=0.92
    )
    format_paired_bar_axis(ax, plot_model_order_reversed_with_gaps)
    ax.set_xlabel("Proportion of fixed-effect variance")
    ax.set_ylabel("")
    for effect_number, container in zip(fixed_effect_numbers, ax.containers):
        ax.bar_label(
            container,
            labels=[
                str(effect_number)
                if base.np.isfinite(bar.get_width()) and bar.get_width() >= 0.025
                else ""
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
        [f"{number}. {name}" for number, name in zip(fixed_effect_numbers, fixed_effect_names)],
        title="Fixed effect",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncols=min(3, len(fixed_effect_names)),
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.3)
    base.savefig(figure_dir / "variance_decomposition.png")

    r2_by_model = fit.set_index("Plot model")["R2"]
    absolute_r2_plot = variance_plot.copy()
    absolute_r2_plot["R2"] = [
        r2_by_model.loc[label]
        for label in absolute_r2_plot.index
    ]
    absolute_r2_plot = absolute_r2_plot[fixed_effect_names].mul(absolute_r2_plot["R2"], axis=0)
    fig, ax = base.plt.subplots(figsize=(13, max(5.5, len(absolute_r2_plot) * 0.9 + 2.5)))
    absolute_r2_plot.reindex(plot_model_order_reversed_with_gaps).plot(
        kind="barh", stacked=True, ax=ax, color=fixed_effect_colors, width=0.92
    )
    format_paired_bar_axis(ax, plot_model_order_reversed_with_gaps)
    ax.set_xlabel("Absolute R²")
    ax.set_ylabel("")
    for effect_number, container in zip(fixed_effect_numbers, ax.containers):
        ax.bar_label(
            container,
            labels=[
                str(effect_number)
                if base.np.isfinite(bar.get_width()) and bar.get_width() >= 0.025
                else ""
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
        [f"{number}. {name}" for number, name in zip(fixed_effect_numbers, fixed_effect_names)],
        title="Fixed effect",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncols=min(3, len(fixed_effect_names)),
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.3)
    base.savefig(figure_dir / "absolute_r2_decomposition.png")
    coefficient_plot = coefficients.pivot_table(index="term", columns="model", values="coef", aggfunc="first")
    base.plt.figure(figsize=(10, max(4, len(coefficient_plot) * 0.3)))
    base.sns.heatmap(coefficient_plot, center=0, cmap="RdBu_r", cbar_kws={"label": "Coefficient"})
    base.plt.xlabel("")
    base.plt.ylabel("Fixed-effect term")
    base.savefig(figure_dir / "coefficient_heatmap.png")
    cards = f"<div class='metric-card'><div class='metric-value'>{len(metrics)}</div><div class='metric-label'>Model fits</div></div>" \
            f"<div class='metric-card'><div class='metric-value'>{int(metrics['nobs'].median()):,}</div><div class='metric-label'>Observations/model</div></div>" \
            f"<div class='metric-card'><div class='metric-value'>{metadata_meta['rows']}</div><div class='metric-label'>Professions</div></div>"
    html = textwrap.dedent(f"""\
        <!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <title>World-Knowledge and Human-Perception Models</title><style>
        body{{margin:0;color:#1d2625;font-family:Georgia,serif;background:#f4efe6}}.page{{max-width:1220px;margin:auto;padding:32px 20px 60px}}
        header,section{{background:#fffaf1;border:1px solid #d9c8ad;border-radius:24px;padding:26px;margin-bottom:22px}}h1{{font-size:clamp(2rem,5vw,4rem);margin:0 0 12px}}h2{{margin:0 0 12px}}.muted{{color:#64706b;line-height:1.5}}
        .metric-grid{{display:flex;gap:12px;margin-top:20px;flex-wrap:wrap}}.metric-card{{border:1px solid #d9c8ad;border-radius:16px;padding:14px;background:#fffdf8;min-width:150px}}.metric-value{{font-size:1.7rem;font-weight:700;color:#315c70}}.metric-label{{color:#64706b}}
        .table-wrap{{overflow-x:auto;border:1px solid #d9c8ad;border-radius:14px;margin:12px 0 18px;background:white}}.data-table{{width:100%;border-collapse:collapse;font-size:.9rem}}.data-table th,.data-table td{{border:1px solid #e2d4bf;padding:8px 10px;text-align:left;white-space:nowrap}}.data-table th{{background:#f0dfc7}}img{{max-width:100%;border:1px solid #d9c8ad;border-radius:14px}}
        </style></head><body><div class="page"><header><h1>World-Knowledge and Human-Perception Models</h1><p class="muted">LassoCV-selected and OLS-refit interaction models for he/she log odds. MalePerc is joined from the first-column profession list in <code>professions.csv</code> and z-scored before fitting; valence and dominance are categorical predictors. Generated on {date.today().isoformat()}.</p><div class="metric-grid">{cards}</div></header>
        <section><h2>Model Fit</h2><div class="table-wrap">{fit_html}</div><img src="figures/r2_comparison.png" alt="R2 comparison"></section>
        <section><h2>R² Change: Human-Perception vs World-Knowledge</h2><p class="muted">Each vertical bar starts at the World-Knowledge model's absolute R² (gray). The attached segment is the change after switching to the Human-Perception model: blue indicates an increase and red a decrease. Labels give the signed ΔR².</p><img src="figures/r2_delta_human_minus_world.png" alt="World-Knowledge R2 with signed change to the Human-Perception model"></section>
        <section><h2>Fixed-Effect Variance</h2><p class="muted">Variance is decomposed across the fixed effects in the Lasso-selected OLS models. Segment numbers in the horizontal bars map to the indexed legend below the plots.</p><div class="table-wrap">{variance_html}</div><img src="figures/variance_decomposition.png" alt="Proportion of fixed-effect variance"><img src="figures/absolute_r2_decomposition.png" alt="Absolute R2 decomposed by fixed effect"></section>
        <section><h2>Selected Coefficients</h2><img src="figures/coefficient_heatmap.png" alt="Coefficient heatmap"><div class="table-wrap">{coefficient_html}</div></section>
        <section><h2>R² Decomposition</h2><p class="muted">Unique R² is estimated from the drop in OLS R² after removing each selected formula term. Lasso alpha is selected by five-fold cross-validation.</p><div class="table-wrap">{decomposition_html}</div></section>
        </div></body></html>""").strip()
    path = report_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    if IMPORT_ERROR is not None:
        print(f"Missing dependencies: {IMPORT_ERROR}", file=sys.stderr)
        return 1
    args = parse_args()
    if args.reuse_existing:
        runs = sorted((args.data_dir / "runs").iterdir())
        metadata_meta = json.loads((args.data_dir / "profession_metadata.json").read_text())
    else:
        metadata, metadata_meta = load_profession_metadata(args.professions)
        args.data_dir.mkdir(parents=True, exist_ok=True)
        (args.data_dir / "profession_metadata.csv").write_text(metadata.to_csv(index=False), encoding="utf-8")
        (args.data_dir / "profession_metadata.json").write_text(json.dumps(metadata_meta, indent=2), encoding="utf-8")
        inputs = discover_inputs(args.results_csv)
        runs = [run_one(path, metadata, metadata_meta, args.data_dir, args.maxiter) for path in base.tqdm(inputs, desc="Input files")]
    if not runs:
        print("No runs found", file=sys.stderr)
        return 1
    report = build_report(runs, args.report_dir, metadata_meta)
    print(f"Wrote report to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
