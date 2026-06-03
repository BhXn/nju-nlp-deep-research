from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and deduplicate Open Track SFT JSONL files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input SFT JSONL files.")
    parser.add_argument("--output", required=True, help="Merged output JSONL.")
    parser.add_argument(
        "--dedupe-by",
        choices=["query_id", "text", "none"],
        default="query_id",
        help="Deduplication key. query_id keeps at most one trajectory per question.",
    )
    parser.add_argument("--min-messages", type=int, default=4, help="Drop rows with too few messages.")
    return parser.parse_args()


def iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def dedupe_key(row: Dict[str, Any], mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "query_id":
        query_id = str(row.get("query_id", "")).strip()
        if query_id:
            return query_id
    return str(row.get("text", "")).strip()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    input_count = 0
    dropped_short = 0
    dropped_duplicate = 0

    for input_path in args.inputs:
        for row in iter_jsonl(input_path):
            input_count += 1
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < args.min_messages:
                dropped_short += 1
                continue

            key = dedupe_key(row, args.dedupe_by)
            if key and key in seen:
                dropped_duplicate += 1
                continue
            if key:
                seen.add(key)
            rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        "Merged {kept}/{total} rows to {output} "
        "(dropped_short={dropped_short}, dropped_duplicate={dropped_duplicate})".format(
            kept=len(rows),
            total=input_count,
            output=output_path,
            dropped_short=dropped_short,
            dropped_duplicate=dropped_duplicate,
        )
    )


if __name__ == "__main__":
    main()
