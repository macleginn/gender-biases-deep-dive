from __future__ import annotations

import argparse
import gzip
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from tqdm import tqdm


def load_professions(path: Path) -> list[tuple[str, re.Pattern[str]]]:
    professions: list[tuple[str, re.Pattern[str]]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        profession = raw.strip()
        if not profession:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(profession)}(?!\w)", re.IGNORECASE)
        professions.append((profession, pattern))
    return professions


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count total occurrences of each profession across the input documents "
            "and the total number of words. Input can be JSONL (with a top-level "
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
        default=Path("professions.txt"),
        help="Text file containing one profession name per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profession_counts.json"),
        help="Output JSON file with aggregate counts.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Process only the first N documents, useful for testing.",
    )

    args = parser.parse_args()
    professions = load_professions(args.professions)
    total_docs = args.max_documents if args.max_documents is not None else count_docs_from_input(args.input)

    profession_counts: Counter[str] = Counter()
    total_words = 0
    processed_docs = 0

    for i, text in enumerate(
        tqdm(iter_texts_from_input(args.input), total=total_docs, desc="Processing documents")
    ):
        if args.max_documents is not None and i >= args.max_documents:
            break

        processed_docs += 1
        total_words += count_words(text)

        for profession, pattern in professions:
            profession_counts[profession] += sum(1 for _ in pattern.finditer(text))

    output = {
        "input": str(args.input),
        "professions": str(args.professions),
        "documents_processed": processed_docs,
        "total_words": total_words,
        "profession_counts": dict(sorted(profession_counts.items())),
    }

    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Processed {processed_docs} documents")
    print(f"Total words: {total_words}")
    print(f"Wrote aggregate counts to {args.output}")


if __name__ == "__main__":
    main()
