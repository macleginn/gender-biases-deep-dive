#!/usr/bin/env python3
"""Plot grouped R² values from the completed model reports.

The plot reads the precomputed ``model_metrics.csv`` files written by
``run_knowledge_perception_profession_report.py`` and
``run_lasso_profession_selected_collocates_report.py``.  The collocates
report contains one row per embedding preprocessing method, so this script
uses its TF-IDF rows.  For every target language model, it places the
Collocates-Based, World-Knowledge, and Human-Perception R² bars next to one
another.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_COLUMNS = {
    "profession_embedding": "Collocates-Based Model",
    "world_knowledge": "World-Knowledge Model",
    "human_perception": "Human-Perception Model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--knowledge-metrics",
        type=Path,
        default=Path("knowledge_perception_model_report/model_metrics.csv"),
        help="Precomputed metrics from the knowledge/perception report.",
    )
    parser.add_argument(
        "--collocates-metrics",
        type=Path,
        default=Path("profession_selected_collocates_report/model_metrics.csv"),
        help="Precomputed metrics from the selected-collocates report.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("combined_model_r2_report"))
    return parser.parse_args()


def clean_model_name(value: str) -> str:
    value = str(value).replace("he_she_odds_results__", "")
    return re.sub(r"__[0-9a-f]{8}$", "", value)


def load_model_labels() -> tuple[list[str], dict[str, str]]:
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
        raise ValueError("full_model_list.txt and model_names.txt must have matching non-empty lines")
    return model_ids, dict(zip(model_ids, model_labels))


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def build_combined_data(knowledge_path: Path, collocates_path: Path) -> pd.DataFrame:
    knowledge = pd.read_csv(knowledge_path)
    profession = pd.read_csv(collocates_path)
    require_columns(knowledge, {"target_model", "model_key", "R2m"}, knowledge_path)
    require_columns(
        profession,
        {"target_model", "embedding_method", "R2m"},
        collocates_path,
    )

    knowledge = knowledge.loc[
        knowledge["model_key"].isin(["world_knowledge", "human_perception"]),
        ["target_model", "model_key", "R2m"],
    ].copy()
    knowledge["target_model"] = knowledge["target_model"].map(clean_model_name)
    knowledge = knowledge.pivot(index="target_model", columns="model_key", values="R2m")
    profession = profession.loc[
        profession["embedding_method"].eq("tfidf"),
        ["target_model", "R2m"],
    ].copy()
    if profession.empty:
        raise ValueError(f"{collocates_path} contains no TF-IDF metrics")
    profession["target_model"] = profession["target_model"].map(clean_model_name)
    duplicate_targets = profession["target_model"].duplicated(keep=False)
    if duplicate_targets.any():
        duplicates = sorted(profession.loc[duplicate_targets, "target_model"].unique())
        raise ValueError(f"{collocates_path} has multiple TF-IDF rows for target models: {duplicates[:10]}")
    profession = profession.set_index("target_model").rename(columns={"R2m": "profession_embedding"})
    combined = profession.join(knowledge, how="inner")
    required_models = list(MODEL_COLUMNS)
    combined = combined.dropna(subset=required_models)
    if combined.empty:
        raise ValueError("No target models have all three precomputed R² values")
    return combined.reset_index()


def main() -> int:
    args = parse_args()
    for path in [args.knowledge_metrics, args.collocates_metrics]:
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
    combined = build_combined_data(args.knowledge_metrics, args.collocates_metrics)
    configured_order, model_labels = load_model_labels()
    available_models = set(combined["target_model"])
    model_order = [name for name in configured_order if name in available_models]
    model_order.extend(sorted(available_models - set(model_order)))
    combined = combined.set_index("target_model").reindex(model_order).reset_index()
    combined.insert(1, "display_model", combined["target_model"].map(lambda name: model_labels.get(name, name)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "combined_r2_data.csv", index=False)

    positions = np.arange(len(combined))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(11, len(combined) * 0.85), 6.5))
    colors = ["#6c8da6", "#d7a84f", "#315c70"]
    for index, (key, label) in enumerate(MODEL_COLUMNS.items()):
        offset = (index - 1) * width
        ax.bar(positions + offset, combined[key], width=width, label=label, color=colors[index])
    ax.set_xticks(positions, combined["display_model"], rotation=45, ha="right")
    ax.set_xlabel("Target language model")
    ax.set_ylabel("Explained variance (R²)")
    ax.set_ylim(0, min(1.0, float(combined[list(MODEL_COLUMNS)].to_numpy().max()) * 1.1))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "combined_r2_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote combined data to {args.output_dir / 'combined_r2_data.csv'}")
    print(f"Wrote plot to {args.output_dir / 'combined_r2_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
