from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_correct_query_ids(eval_path: str | Path, include_uncertain: bool = False) -> Set[str]:
    correct: Set[str] = set()
    for row in load_jsonl(eval_path):
        if row.get("type") == "summary":
            continue
        judgment = str(row.get("eval_judgment", "")).upper()
        if judgment == "CORRECT" or (include_uncertain and judgment == "UNCERTAIN"):
            correct.add(str(row.get("query_id", "")))
    return correct


def compact_messages(messages: Iterable[Dict[str, Any]], max_tool_chars: int) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if item.get("role") == "tool" and isinstance(item.get("content"), str):
            content = item["content"]
            if len(content) > max_tool_chars:
                item["content"] = content[: max_tool_chars - 3].rstrip() + "..."
        compacted.append(item)
    return compacted


def messages_to_text(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}")
        if message.get("tool_calls"):
            parts.append(json.dumps({"tool_calls": message["tool_calls"]}, ensure_ascii=False))
        parts.append("<|im_end|>")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT data from successful Deep Research trajectories.")
    parser.add_argument("--submission", required=True, help="Agent submission JSONL.")
    parser.add_argument("--eval-results", required=True, help="Evaluation JSONL produced by agent.eval.")
    parser.add_argument(
        "--source-split",
        required=True,
        choices=("train", "dev", "test"),
        help="Data split used to produce the submission/eval files. SFT from test data is refused.",
    )
    parser.add_argument("--output", default="open_track/data/sft_success.jsonl", help="Output SFT JSONL.")
    parser.add_argument("--max-tool-chars", type=int, default=3000, help="Compact long tool outputs.")
    parser.add_argument(
        "--include-uncertain",
        action="store_true",
        help="Also include trajectories judged UNCERTAIN. Default keeps only CORRECT.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_split == "test":
        raise ValueError(
            "Refusing to build SFT data from a test split. "
            "Use only training/development trajectories with legitimate labels."
        )
    correct_ids = load_correct_query_ids(args.eval_results, include_uncertain=args.include_uncertain)
    submissions = load_jsonl(args.submission)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for record in submissions:
            query_id = str(record.get("query_id", ""))
            if query_id not in correct_ids:
                continue
            messages = compact_messages(record.get("messages", []), max_tool_chars=args.max_tool_chars)
            row = {
                "query_id": query_id,
                "messages": messages,
                "text": messages_to_text(messages),
                "predicted_answer": record.get("predicted_answer", ""),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Saved {kept} SFT rows to {output_path}")


if __name__ == "__main__":
    main()
