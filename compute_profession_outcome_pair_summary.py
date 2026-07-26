#!/usr/bin/env python3
"""Summarize extracted profession/outcome pairs by target model."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("profession_outcome_pairs.csv"),
        help="CSV containing the already extracted pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profession_outcome_pair_summary.csv"),
        help="Output summary CSV.",
    )
    parser.add_argument(
        "--modelling-data-dir",
        type=Path,
        default=Path("modelling_data"),
        help="Directory containing the original per-model CSV files.",
    )
    return parser.parse_args()


def total_pairs(path: Path, keys: tuple[str, ...]) -> tuple[str, int]:
    """Return the model tag and total unordered pairs sharing ``keys``."""
    groups: Counter[tuple[str, ...]] = Counter()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(keys) | {"model_tag"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        model_tag = None
        for row in reader:
            model_tag = model_tag or row["model_tag"]
            groups[tuple(row[key] for key in keys)] += 1

    if model_tag is None:
        raise ValueError(f"{path} contains no data rows")
    return model_tag, sum(n * (n - 1) // 2 for n in groups.values())


def load_total_pair_counts(data_dir: Path) -> dict[str, dict[str, int]]:
    totals: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for path in sorted(data_dir.glob("he_she_odds_results__*.csv")):
        model, profession_pairs = total_pairs(path, ("profession",))
        _, profession_verb_pairs = total_pairs(path, ("profession", "verb"))
        totals[model]["total_same_profession_pairs"] = profession_pairs
        totals[model]["total_same_profession_and_verb_pairs"] = profession_verb_pairs
    return dict(totals)


def main() -> None:
    args = parse_args()
    counts: Counter[tuple[str, str]] = Counter()

    with args.input.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pair_type", "example_1_model_tag"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{args.input} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            counts[(row["example_1_model_tag"], row["pair_type"])] += 1

    total_counts = load_total_pair_counts(args.modelling_data_dir)
    models = sorted({model for model, _ in counts} | set(total_counts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_model",
        "same_profession_pairs",
        "same_profession_and_verb_pairs",
        "total_same_profession_pairs",
        "total_same_profession_and_verb_pairs",
        "total_pairs",
    ]

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in models:
            same_profession = counts[(model, "same_profession")]
            same_profession_and_verb = counts[
                (model, "same_profession_and_verb")
            ]
            totals = total_counts.get(model, {})
            total_same_profession = totals.get("total_same_profession_pairs", 0)
            total_same_profession_and_verb = totals.get(
                "total_same_profession_and_verb_pairs", 0
            )
            writer.writerow({
                "target_model": model,
                "same_profession_pairs": same_profession,
                "same_profession_and_verb_pairs": same_profession_and_verb,
                "total_same_profession_pairs": total_same_profession,
                "total_same_profession_and_verb_pairs": total_same_profession_and_verb,
                "total_pairs": same_profession + same_profession_and_verb,
            })

    print(f"Wrote summary for {len(models)} models to {args.output}")


if __name__ == "__main__":
    main()
