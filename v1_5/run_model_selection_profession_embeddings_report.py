#!/usr/bin/env python3
"""Fit and report models that represent profession with SVD embeddings.

The profession embedding is computed once from profession_collocates.csv and
joined to each modelling data set.  The embedding dimensions are fixed effects,
not random effects.  Backward selection is performed only over the ordinary
fixed effects, so embedding dimensions are never introduced into interactions.

R-style model formula (with k=5):
    log_he_she_odds ~ (tense + semantic_role + syntactic_role + valence +
                       dominance + log(frequency) + lex_emb_norm)^2 +
                       profession_embedding_1 + ... + profession_embedding_5
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
    from scipy.stats import chi2
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


REQUIRED_COLUMNS = {
    "tense", "syntactic_role", "semantic_role", "valence", "dominance",
    "frequency", "lex_emb_norm", "profession", "log_he_she_odds",
}
CATEGORICAL_COLUMNS = {
    "tense", "syntactic_role", "semantic_role", "valence", "dominance", "profession",
}
FIXED_PREDICTORS = [
    "tense", "semantic_role", "syntactic_role", "valence", "dominance",
    "frequency", "lex_emb_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path, nargs="*")
    parser.add_argument("--profession-collocates", type=Path, default=Path("profession_collocates.csv"))
    parser.add_argument("--k", type=int, default=5, help="Number of SVD embedding dimensions (default: 5).")
    parser.add_argument("--data-dir", type=Path, default=Path("profession_embedding_model_selection_data"))
    parser.add_argument("--report-dir", type=Path, default=Path("profession_embedding_model_selection_report"))
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--starting-fixed-effect-interactions", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def clean_model_name(value: str) -> str:
    return re.sub(r"__[0-9a-f]{8}$", "", str(value).replace("he_she_odds_results__", ""))


def term(name: str) -> str:
    return "log_frequency" if name == "frequency" else (f"C({name})" if name in CATEGORICAL_COLUMNS else name)


def fixed_terms(predictors: list[str], interactions: bool) -> list[str]:
    main = [term(name) for name in predictors]
    return main + ([f"{a}:{b}" for a, b in combinations(main, 2)] if interactions else [])


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
    return df.dropna(subset=sorted(REQUIRED_COLUMNS)).copy()


def compute_profession_embeddings(path: Path, k: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """TF-IDF transform collocate counts and return k SVD coordinates per profession."""
    if k < 1:
        raise ValueError("k must be at least 1")
    collocates = pd.read_csv(path)
    if collocates.shape[1] < 2:
        raise ValueError(f"{path} must contain a profession column and collocate columns")
    profession_column = collocates.columns[0]
    professions = collocates[profession_column].astype("string").str.strip()
    if professions.isna().any() or professions.duplicated().any():
        raise ValueError(f"{path} contains missing or duplicate profession names")
    counts = collocates.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if (counts < 0).any().any():
        raise ValueError(f"{path} contains negative collocate counts")
    if len(collocates) < 2:
        raise ValueError("At least two professions are required for SVD")
    tfidf = TfidfTransformer().fit_transform(counts.to_numpy(dtype=float))
    max_components = min(tfidf.shape[0] - 1, tfidf.shape[1])
    if k > max_components:
        raise ValueError(f"k={k} exceeds the maximum available SVD components ({max_components})")
    coordinates = TruncatedSVD(n_components=k, random_state=0).fit_transform(tfidf)
    columns = [f"profession_embedding_{index}" for index in range(1, k + 1)]
    embedding = pd.DataFrame(coordinates, columns=columns)
    embedding.insert(0, "profession", professions.to_numpy())
    return embedding, {
        "source": str(path.resolve()), "profession_column": str(profession_column),
        "rows": int(len(embedding)), "collocate_columns": int(counts.shape[1]),
        "k": k, "columns": columns,
    }


def add_embedding(df: pd.DataFrame, embedding: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["profession"] = result["profession"].astype("string").str.strip()
    merged = result.merge(embedding, on="profession", how="left", validate="many_to_one")
    missing = int(merged[embedding.columns[1]].isna().sum())
    if missing:
        unknown = sorted(result.loc[merged[embedding.columns[1]].isna(), "profession"].unique())
        raise ValueError(f"{missing} rows have professions absent from the embedding: {unknown[:10]}")
    return merged


def fixed_variances(result: Any, embedding_columns: list[str]) -> dict[str, float]:
    info = result.model.data.design_info
    design = np.asarray(result.model.exog)
    beta = np.asarray(result.params)
    values: dict[str, float] = {}
    embedding_slices = [info.term_name_slices[c] for c in embedding_columns if c in info.term_name_slices]
    if embedding_slices:
        embedding_design = np.concatenate([design[:, sl] for sl in embedding_slices], axis=1)
        embedding_beta = np.concatenate([beta[sl] for sl in embedding_slices])
        values["fixed_effect_variance_profession_embedding"] = float(
            np.var(embedding_design @ embedding_beta, ddof=1)
        )
    for name, sl in info.term_name_slices.items():
        if name in embedding_columns:
            continue
        values[f"fixed_effect_variance_{name}"] = float(np.var(design[:, sl] @ beta[sl], ddof=1))
    return values


def fit_model(df: pd.DataFrame, name: str, terms: list[str], embedding_columns: list[str], run_dir: Path) -> dict[str, Any]:
    model_dir = run_dir / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    formula = "log_he_she_odds ~ " + " + ".join(terms)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = smf.ols(formula=formula, data=df).fit()
        fixed = pd.DataFrame({"term": result.params.index, "coef": result.params, "std_err": result.bse,
                              "t": result.tvalues, "p_value": result.pvalues,
                              "ci_low": result.conf_int()[0], "ci_high": result.conf_int()[1]})
        fixed.to_csv(model_dir / "fixed_effects.csv", index=False)
        metrics = {
            "formula": formula, "model_type": "OLS", "optimizer": "closed_form",
            "converged": True, "nobs": int(result.nobs), "aic": float(result.aic),
            "bic": float(result.bic), "log_likelihood": float(result.llf),
            "residual_variance": float(result.scale), "R2m": float(result.rsquared),
            "R2c": float(result.rsquared),
            "warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
            **fixed_variances(result, embedding_columns),
        }
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (model_dir / "summary.txt").write_text(result.summary().as_text(), encoding="utf-8")
        return {"status": "ok", "result": result, "metrics": metrics, "terms": terms, "name": name}
    except Exception as exc:
        (model_dir / "fit_failed.json").write_text(json.dumps({"error": repr(exc)}, indent=2), encoding="utf-8")
        return {"status": "failed", "error": repr(exc), "name": name, "terms": terms}


def select_model(df: pd.DataFrame, run_dir: Path, embedding_columns: list[str], alpha: float,
                 interactions: bool) -> tuple[dict[str, Any], list[str], pd.DataFrame]:
    ordinary = fixed_terms(FIXED_PREDICTORS, interactions)
    all_terms = ordinary + embedding_columns
    current = fit_model(df, "selected_profession_embedding__full", all_terms, embedding_columns, run_dir)
    trace: list[dict[str, Any]] = []
    if current["status"] != "ok":
        return current, all_terms, pd.DataFrame()
    step = 0
    while True:
        candidates = []
        for candidate in ordinary:
            # Preserve hierarchy: a main effect cannot be dropped while one of
            # its interactions remains in the current model.
            if ":" not in candidate and any(
                candidate in other.split(":") for other in current["terms"] if ":" in other
            ):
                continue
            reduced = [t for t in current["terms"] if t != candidate]
            if len(reduced) == len(current["terms"]):
                continue
            fit = fit_model(df, f"selected_profession_embedding__candidate__{step}__{slugify(candidate)}",
                            reduced, embedding_columns, run_dir)
            if fit["status"] != "ok":
                continue
            lr = 2 * (current["result"].llf - fit["result"].llf)
            df_diff = max(1, int(current["result"].df_model - fit["result"].df_model))
            p_value = float(chi2.sf(lr, df_diff))
            row = {"step": step, "candidate_term": candidate, "candidate_model": fit["name"],
                   "lr_stat": float(lr), "df_diff": df_diff, "p_value": p_value,
                   **fit["metrics"]}
            trace.append(row)
            candidates.append((p_value, candidate, fit))
        if not candidates:
            break
        best_p, best_term, best_fit = max(candidates, key=lambda x: x[0])
        if not np.isfinite(best_p) or best_p <= alpha:
            break
        current = best_fit
        ordinary.remove(best_term)
        step += 1
    selected = fit_model(df, "selected_profession_embedding", current["terms"], embedding_columns, run_dir)
    trace_df = pd.DataFrame(trace)
    if not trace_df.empty:
        trace_df.to_csv(run_dir / "model_selection_trace.csv", index=False)
    (run_dir / "selected_model.json").write_text(json.dumps({"selected_model": selected["name"],
        "selected_terms": selected["terms"], "alpha": alpha}, indent=2), encoding="utf-8")
    return selected, selected["terms"], trace_df


def run_one(input_csv: Path, embedding: pd.DataFrame, embedding_meta: dict[str, Any], data_dir: Path,
            maxiter: int, alpha: float, interactions: bool) -> Path:
    run_dir = data_dir / "runs" / slugify(input_csv.stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_csv, data_dir / "inputs" / input_csv.name)
    df = add_embedding(load_results_csv(input_csv), embedding)
    df.to_csv(run_dir / "prepared_results.csv", index=False)
    embedding_columns = list(embedding.columns[1:])
    selected, terms, trace = select_model(df, run_dir, embedding_columns, alpha, interactions)
    baseline = fit_model(df, "no_profession_embedding_baseline", fixed_terms(FIXED_PREDICTORS, interactions),
                         embedding_columns, run_dir)
    (run_dir / "run_summary.json").write_text(json.dumps({"input_csv": str(input_csv.resolve()),
        "embedding": embedding_meta, "selected_terms": terms, "selection_steps": len(trace),
        "selected_model": selected["name"], "baseline_model": baseline["name"]}, indent=2), encoding="utf-8")
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


def build_report(run_dirs: list[Path], report_dir: Path, embedding_meta: dict[str, Any]) -> Path:
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
                rows.append({"target_model": clean_model_name(Path(summary["input_csv"]).stem),
                             "model_name": model_dir.name, "model_dir": str(model_dir),
                             **json.loads(metrics_path.read_text())})
    all_metrics = pd.DataFrame(rows)
    all_metrics.to_csv(report_dir / "all_model_metrics.csv", index=False)
    selected = all_metrics.loc[
        all_metrics["model_name"] == "selected_profession_embedding"
    ].copy()
    selected.to_csv(report_dir / "model_metrics.csv", index=False)

    if selected.empty:
        raise ValueError("No selected profession-embedding models were found in the supplied runs")

    selected["target_model"] = selected["target_model"].map(clean_model_name)
    selected = selected.sort_values("target_model").reset_index(drop=True)
    fixed_rows = []
    for _, row in selected.iterrows():
        fixed = pd.read_csv(Path(row["model_dir"]) / "fixed_effects.csv")
        fixed.insert(0, "target_model", row["target_model"])
        fixed_rows.append(fixed)
    coefficients = pd.concat(fixed_rows, ignore_index=True) if fixed_rows else pd.DataFrame()
    coefficients.to_csv(report_dir / "selected_coefficients.csv", index=False)
    variance_cols = [c for c in selected.columns if c.startswith("fixed_effect_variance_")]
    variance = selected[["target_model", *variance_cols]].copy()
    variance = variance.rename(columns={
        "target_model": "Model",
        "fixed_effect_variance_profession_embedding": "Profession embedding",
        **{
            column: column.removeprefix("fixed_effect_variance_").replace("_", " ")
            for column in variance_cols
            if column != "fixed_effect_variance_profession_embedding"
        },
    })
    variance.to_csv(report_dir / "variance_decomposition.csv", index=False)

    fit_table = selected[
        ["target_model", "converged", "nobs", "aic", "bic", "log_likelihood", "R2m", "residual_variance"]
    ].rename(columns={
        "target_model": "Model", "converged": "Converged", "nobs": "N", "aic": "AIC",
        "bic": "BIC", "log_likelihood": "Log likelihood", "R2m": "R2", "residual_variance": "Residual variance",
    })
    fit_html = write_table(fit_table, table_dir / "selected_model_fit.html")
    variance_html = write_table(variance, table_dir / "variance_decomposition.html")
    coefficient_html = write_table(coefficients, table_dir / "selected_coefficients.html", classes="data-table compact")

    sns.set_theme(style="whitegrid", context="notebook")
    model_order = selected["target_model"].tolist()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=fit_table, x="Model", y="R2", order=model_order, color="#315c70")
    plt.ylim(0, max(1.0, float(fit_table["R2"].max()) * 1.08))
    plt.xticks(rotation=25, ha="right")
    plt.xlabel("")
    plt.ylabel("Explained variance (R2)")
    savefig(figure_dir / "r2_comparison.png")

    variance_plot = variance.set_index("Model")
    variance_plot = variance_plot.div(variance_plot.sum(axis=1).replace(0, np.nan), axis=0)
    fixed_effect_names = variance_plot.columns.tolist()
    fixed_effect_numbers = list(range(1, len(fixed_effect_names) + 1))
    fixed_effect_colors = sns.color_palette("husl", n_colors=len(fixed_effect_names))
    fig, ax = plt.subplots(figsize=(13, max(5.5, len(model_order) * 0.9 + 2.5)))
    variance_plot.loc[model_order].plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=fixed_effect_colors,
    )
    ax.set_xlabel("Proportion of fixed-effect variance")
    ax.set_ylabel("")
    for effect_number, container in zip(fixed_effect_numbers, ax.containers):
        ax.bar_label(
            container,
            labels=[
                str(effect_number)
                if np.isfinite(bar.get_width()) and bar.get_width() >= 0.025
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
    savefig(figure_dir / "variance_decomposition.png")

    coefficient_plot = coefficients.pivot(index="term", columns="target_model", values="coef").reindex(columns=model_order)
    plt.figure(figsize=(10, max(4, 0.35 * len(coefficient_plot))))
    sns.heatmap(coefficient_plot, center=0, cmap="RdBu_r", cbar_kws={"label": "Coefficient"})
    plt.xlabel("")
    plt.ylabel("Fixed-effect term")
    savefig(figure_dir / "coefficient_heatmap.png")

    cards = [
        ("Target models", str(len(selected))),
        ("Embedding dimensions", str(embedding_meta["k"])),
        ("Observations/model", f"{int(selected['nobs'].median()):,}"),
        ("Selected models converged", "yes" if bool(selected["converged"].all()) else "no"),
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
            .table-wrap {{ overflow-x:auto; border:1px solid var(--border); border-radius:14px; margin:12px 0 18px; background:white; }} .data-table {{ width:100%; border-collapse:collapse; font-size:.92rem; }}
            .data-table th,.data-table td {{ border:1px solid #e2d4bf; padding:8px 10px; text-align:left; vertical-align:top; white-space:nowrap; }} .data-table th {{ background:#f0dfc7; }} .compact {{ font-size:.84rem; }}
            @media (max-width:760px) {{ .page {{ padding:14px 10px 36px; }} header,section {{ padding:16px; }} .figure-grid {{ grid-template-columns:1fr; }} }}
          </style>
        </head>
        <body><div class="page">
          <header><h1>Profession-Embedding Model Selection Report</h1>
            <p class="subtitle">Backward-selected OLS models for log he/she odds. Profession is represented by a {escape(str(embedding_meta['k']))}-dimension SVD embedding. Generated on {date.today().isoformat()}.</p>
            <div class="metric-grid">{cards_html}</div>
          </header>
          <section><h2>Selected Model Fit</h2><p class="section-note">One final selected configuration is reported for each target model. All candidate fits remain available in <code>all_model_metrics.csv</code>.</p><div class="table-wrap">{fit_html}</div>{figure("figures/r2_comparison.png", "Explained variance (R2) by selected model.")}</section>
          <section><h2>Fixed-Effect Variance</h2><p class="section-note">The embedding dimensions are grouped as one predictor; they are not included in interactions. Segment numbers in the horizontal bars map to the indexed legend below the plot.</p><div class="table-wrap">{variance_html}</div>{figure("figures/variance_decomposition.png", "Proportion of fixed-effect variance by selected model; segment numbers map to the indexed legend.")}</section>
          <section><h2>Selected Coefficients</h2>{figure("figures/coefficient_heatmap.png", "Fixed-effect coefficients by selected model.")}<div class="table-wrap">{coefficient_html}</div></section>
        </div></body></html>
        """
    ).strip()
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
        meta = json.loads((runs[0] / "run_summary.json").read_text())["embedding"] if runs else {"k": args.k, "source": str(args.profession_collocates)}
    else:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        (args.data_dir / "inputs").mkdir(exist_ok=True)
        inputs = discover_inputs(args.results_csv)
        embedding, meta = compute_profession_embeddings(args.profession_collocates, args.k)
        (args.data_dir / "embedding.csv").write_text(embedding.to_csv(index=False), encoding="utf-8")
        (args.data_dir / "embedding_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        runs = [run_one(p, embedding, meta, args.data_dir, args.maxiter, args.alpha, args.starting_fixed_effect_interactions) for p in tqdm(inputs, desc="Input files")]
    if not runs:
        print("No runs found", file=sys.stderr)
        return 1
    report = build_report(runs, args.report_dir, meta)
    print(f"Wrote report to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
