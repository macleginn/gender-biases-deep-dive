#!/usr/bin/env python3
"""Find pairs of model examples whose binary outcomes disagree.

For each CSV in ``modelling_data`` this script writes every unordered pair
that either:

* has the same profession; or
* has the same profession and verb.

The binary outcome is ``he`` when the log odds are positive and ``she``
otherwise.  The input files currently call this column
``log_he_she_odds``; ``he_she_log_odds`` is accepted as an alias.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Iterable


PAIR_TYPES = (
    ("same_profession", ("profession",)),
    ("same_profession_and_verb", ("profession", "verb")),
)
REQUIRED_COLUMNS = {"profession", "verb"}
LOG_ODDS_COLUMNS = ("he_she_log_odds", "log_he_she_odds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("modelling_data"),
        help="Directory containing model CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profession_outcome_pairs.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def discover_inputs(input_dir: Path) -> list[Path]:
    inputs = sorted(input_dir.glob("he_she_odds_results__*.csv"))
    if not inputs:
        raise FileNotFoundError(f"No model CSV files found in {input_dir}")
    return inputs


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        columns = list(reader.fieldnames)
        missing = REQUIRED_COLUMNS - set(columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        log_odds_column = next((name for name in LOG_ODDS_COLUMNS if name in columns), None)
        if log_odds_column is None:
            raise ValueError(
                f"{path} must contain one of the log-odds columns: {LOG_ODDS_COLUMNS}"
            )

        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                log_odds = float(row[log_odds_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{row_number} has a non-numeric {log_odds_column!r}"
                ) from exc
            row = {key: (value or "").strip() for key, value in row.items()}
            row["_binary_outcome"] = "he" if log_odds > 0 else "she"
            row["_source_row"] = str(row_number)
            rows.append(row)
    return rows, columns


def iter_pairs(
    rows: list[dict[str, str]], pair_type: str, keys: tuple[str, ...]
) -> Iterable[tuple[dict[str, str], dict[str, str]]]:
    groups: dict[tuple[str, ...], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"he": [], "she": []}
    )
    for row in rows:
        groups[tuple(row[key] for key in keys)][row["_binary_outcome"]].append(row)

    for group in groups.values():
        # Cross-product gives every unordered pair exactly once because the
        # two outcome buckets are distinct.
        yield from product(group["she"], group["he"])


def write_pairs(output: Path, inputs: list[Path]) -> tuple[int, dict[str, int]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    counts: dict[str, int] = defaultdict(int)
    fieldnames: list[str] | None = None

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = None
        for input_path in inputs:
            rows, source_columns = load_rows(input_path)
            if fieldnames is None:
                fieldnames = ["model_file", "pair_type", "profession", "verb"]
                fieldnames += [f"example_1_{column}" for column in source_columns]
                fieldnames += ["example_1_binary_outcome", "example_1_source_row"]
                fieldnames += [f"example_2_{column}" for column in source_columns]
                fieldnames += ["example_2_binary_outcome", "example_2_source_row"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()

            assert writer is not None
            for pair_type, keys in PAIR_TYPES:
                for first, second in iter_pairs(rows, pair_type, keys):
                    result = {
                        "model_file": input_path.name,
                        "pair_type": pair_type,
                        "profession": first["profession"],
                        "verb": first["verb"] if "verb" in keys else "",
                    }
                    for prefix, row in (("example_1_", first), ("example_2_", second)):
                        result.update({f"{prefix}{column}": row[column] for column in source_columns})
                        result[f"{prefix}binary_outcome"] = row["_binary_outcome"]
                        result[f"{prefix}source_row"] = row["_source_row"]
                    writer.writerow(result)
                    total += 1
                    counts[pair_type] += 1

    return total, dict(counts)


def main() -> None:
    args = parse_args()
    inputs = discover_inputs(args.input_dir)
    total, counts = write_pairs(args.output, inputs)
    print(f"Processed {len(inputs)} model files and wrote {total} pairs to {args.output}")
    for pair_type, count in counts.items():
        print(f"  {pair_type}: {count}")


if __name__ == "__main__":
    main()
