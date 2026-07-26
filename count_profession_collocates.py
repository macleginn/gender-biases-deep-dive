from __future__ import annotations

import argparse
import gzip
import csv
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from tqdm import tqdm


def load_professions(path: Path) -> list[tuple[str, re.Pattern[str]]]:
    professions: list[tuple[str, re.Pattern[str]]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle)
        next(rows, None)  # Header row.
        for row in rows:
            if not row:
                continue
            profession = row[0].strip()
            if not profession:
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(profession)}(?!\w)", re.IGNORECASE)
            professions.append((profession, pattern))
    return professions


TOKEN_PATTERN = re.compile(r"\b\w+(?:'\w+)?\b")


def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def iter_profession_window_collocates(
    text: str,
    professions: list[tuple[str, re.Pattern[str]]],
    left_window_size: int,
    right_window_size: int,
) -> dict[str, Counter[str]]:
    tokens = tokenize_with_spans(text)
    if not tokens:
        return {}

    lowered_tokens = [token.lower() for token, _, _ in tokens]
    token_starts = [start for _, start, _ in tokens]
    token_ends = [end for _, _, end in tokens]
    collocates: dict[str, Counter[str]] = {profession: Counter() for profession, _ in professions}

    for profession, pattern in professions:
        for match in pattern.finditer(text):
            match_start, match_end = match.span()
            start_token = 0
            while start_token < len(tokens) and token_ends[start_token] <= match_start:
                start_token += 1
            end_token = start_token
            while end_token < len(tokens) and token_starts[end_token] < match_end:
                end_token += 1

            if start_token >= len(tokens):
                continue

            window_start = max(0, start_token - left_window_size)
            window_end = min(len(tokens), end_token + right_window_size)

            for j in range(window_start, window_end):
                if start_token <= j < end_token:
                    continue
                collocates[profession][lowered_tokens[j]] += 1

    return collocates


def iter_texts(jsonl_path: Path):
    opener = gzip.open if jsonl_path.suffix.lower() == ".gz" else Path.open
    with opener(jsonl_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj, str):
                    obj = json.loads(obj)
                if isinstance(obj, dict):
                    text = obj.get("text")
                    if isinstance(text, str) and text:
                        yield text
                        continue
                    raw = obj.get("raw_json")
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    if isinstance(raw, str) and raw:
                        try:
                            inner = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        text = inner.get("text") if isinstance(inner, dict) else None
                        if isinstance(text, str) and text:
                            yield text


def count_texts(jsonl_path: Path) -> int:
    opener = gzip.open if jsonl_path.suffix.lower() == ".gz" else Path.open
    with opener(jsonl_path, "rt", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _sqlite_connect_immutable(db_path: Path) -> sqlite3.Connection:
    # Some network/distributed filesystems don't support SQLite locking well.
    # `immutable=1` tells SQLite to never try to write/lock, and treat the DB as read-only.
    uri = f"file:{db_path.resolve()}?immutable=1"
    return sqlite3.connect(uri, uri=True)


def iter_texts_sqlite(
    db_path: Path,
    *,
    table: str = "reservoir",
    raw_json_column: str = "raw_json",
    text_key: str = "text",
    fetch_size: int = 1000,
):
    con = _sqlite_connect_immutable(db_path)
    try:
        cur = con.cursor()
        cur.execute(f"SELECT {raw_json_column} FROM {table}")
        while True:
            rows = cur.fetchmany(fetch_size)
            if not rows:
                break
            for (raw,) in rows:
                if raw is None:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                text = obj.get(text_key)
                if isinstance(text, str) and text:
                    yield text
    finally:
        con.close()


def count_texts_sqlite(
    db_path: Path, *, table: str = "reservoir"
) -> int:
    con = _sqlite_connect_immutable(db_path)
    try:
        cur = con.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        (n,) = cur.fetchone()
        return int(n)
    finally:
        con.close()


def _is_sqlite_path(path: Path) -> bool:
    return path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}


def iter_texts_from_input(path: Path):
    if _is_sqlite_path(path):
        yield from iter_texts_sqlite(path)
    else:
        yield from iter_texts(path)


def count_docs_from_input(path: Path) -> int:
    if _is_sqlite_path(path):
        return count_texts_sqlite(path)
    return count_texts(path)


def count_words(text: str) -> int:
    return sum(1 for _ in re.finditer(r"\S+", text))


def write_collocate_csv(
    output_path: Path,
    profession_collocates: dict[str, Counter[str]],
    profession_order: list[str],
) -> None:
    column_totals: Counter[str] = Counter()
    for counts in profession_collocates.values():
        column_totals.update(counts)

    columns = sorted(
        word for word, total in column_totals.items() if total >= 5
    )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["profession", *columns])
        for profession in profession_order:
            row = [profession]
            counts = profession_collocates.get(profession, Counter())
            row.extend(counts.get(word, 0) for word in columns)
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count collocate words in a configurable token window around each target profession "
            "across the input documents. Input can be JSONL (with a top-level "
            "\"text\" field; .jsonl.gz is supported) or a SQLite database "
            "(reads reservoir.raw_json[\"text\"])."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs_w_professions.jsonl"),
        help=(
            "Input JSONL/JSONL.GZ (documents with a top-level text field, or a top-level raw_json "
            "string containing JSON with a text field) or SQLite .db/.sqlite/.sqlite3 "
            "(reads reservoir.raw_json JSON and extracts the text field)."
        ),
    )
    parser.add_argument(
        "--professions",
        type=Path,
        default=Path("professions.csv"),
        help="CSV file whose first column contains profession names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profession_collocates.csv"),
        help="Output CSV file with collocate counts (professions as rows).",
    )
    parser.add_argument(
        "--left-window-size",
        type=int,
        default=100,
        help="Number of tokens to include to the left of a profession mention.",
    )
    parser.add_argument(
        "--right-window-size",
        type=int,
        default=100,
        help="Number of tokens to include to the right of a profession mention.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Process only the first N documents, useful for testing.",
    )

    args = parser.parse_args()
    professions = load_professions(args.professions)
    profession_order = [profession for profession, _ in professions]
    total_docs = args.max_documents if args.max_documents is not None else count_docs_from_input(args.input)

    total_words = 0
    processed_docs = 0
    profession_collocates: dict[str, Counter[str]] = {
        profession: Counter() for profession in profession_order
    }

    for i, text in enumerate(
        tqdm(iter_texts_from_input(args.input), total=total_docs, desc="Processing documents")
    ):
        if args.max_documents is not None and i >= args.max_documents:
            break

        processed_docs += 1
        total_words += count_words(text)

        collocates = iter_profession_window_collocates(
            text,
            professions,
            args.left_window_size,
            args.right_window_size,
        )
        for profession, counts in collocates.items():
            profession_collocates[profession].update(counts)

    write_collocate_csv(args.output, profession_collocates, profession_order)

    print(f"Processed {processed_docs} documents")
    print(f"Total words: {total_words}")
    print(f"Wrote collocate table to {args.output}")


if __name__ == "__main__":
    main()
