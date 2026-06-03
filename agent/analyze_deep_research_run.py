from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a Deep Research submission/evaluation pair.")
    parser.add_argument("--submission", required=True, help="Submission JSONL produced by agent.run_deep_research.")
    parser.add_argument("--eval", required=True, help="Evaluation JSONL produced by agent.eval.")
    parser.add_argument("--dataset", default="browsecomp_plus_hard50.jsonl", help="Dataset JSONL with gold answers.")
    parser.add_argument("--output", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--top", type=int, default=20, help="Maximum examples per printed section.")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def query_id(row: Dict[str, Any]) -> str:
    return str(row.get("query_id") or row.get("id") or row.get("qid") or row.get("question_id") or "")


def normalize(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def compact(text: Any, max_chars: int = 120) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def tool_call_names(messages: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    names: Dict[str, str] = {}
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            call_id = tool_call.get("id")
            name = (tool_call.get("function") or {}).get("name")
            if call_id and name:
                names[str(call_id)] = str(name)
    return names


def decode_json_payload(payload: Any) -> Any:
    if not isinstance(payload, str):
        return payload
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload


def iter_tool_results(messages: Iterable[Dict[str, Any]]) -> Iterable[Tuple[str, Any, str]]:
    id_to_name = tool_call_names(messages)
    for message in messages:
        if message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        name = id_to_name.get(call_id, "unknown_tool")
        content = message.get("content", "")
        yield name, decode_json_payload(content), content if isinstance(content, str) else json.dumps(content)


def is_unknown_like(prediction: Any) -> bool:
    return bool(
        re.search(
            r"\b(?:unknown|unable|not enough|cannot determine|insufficient evidence|none)\b",
            str(prediction or ""),
            flags=re.IGNORECASE,
        )
    )


def analyze_run(submission_path: str, eval_path: str, dataset_path: str) -> Dict[str, Any]:
    dataset = {query_id(row): row for row in load_jsonl(dataset_path)}
    submissions = {query_id(row): row for row in load_jsonl(submission_path)}
    eval_rows = load_jsonl(eval_path)
    eval_summary = next((row for row in eval_rows if row.get("type") == "summary"), {})
    eval_items = [row for row in eval_rows if row.get("type") != "summary"]

    categories: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    lexical_verdicts: Counter[str] = Counter()
    model_verdicts: Counter[str] = Counter()
    examples: List[Dict[str, Any]] = []

    for item in eval_items:
        qid = query_id(item)
        submission = submissions.get(qid, {})
        dataset_row = dataset.get(qid, {})
        gold = item.get("gold_answer") or dataset_row.get("answer")
        pred = item.get("predicted_answer") or submission.get("predicted_answer")
        correct = item.get("eval_judgment") == "CORRECT" or item.get("correct") is True
        status = item.get("status") or submission.get("status") or ""
        statuses[status] += 1

        tool_texts: List[str] = []
        lexical_verify_count = 0
        model_verify_count = 0
        supported_model_verify = 0
        missing_docid_calls = 0
        for name, result, raw_content in iter_tool_results(submission.get("messages") or []):
            tool_counts[name] += 1
            tool_texts.append(raw_content)
            if name == "verify_claim" and isinstance(result, dict):
                if "supported_likely" in result:
                    lexical_verify_count += 1
                    lexical_verdicts[str(bool(result.get("supported_likely")))] += 1
                    if result.get("missing_docids"):
                        missing_docid_calls += 1
                if "verdict" in result:
                    model_verify_count += 1
                    verdict = str(result.get("verdict"))
                    model_verdicts[verdict] += 1
                    if verdict == "supported":
                        supported_model_verify += 1

        all_tool_text = normalize("\n".join(tool_texts))
        gold_norm = normalize(gold)
        pred_norm = normalize(pred)
        gold_seen = bool(gold_norm and len(gold_norm) >= 3 and gold_norm in all_tool_text)
        pred_seen = bool(pred_norm and len(pred_norm) >= 3 and pred_norm in all_tool_text)

        if correct:
            category = "correct"
        elif is_unknown_like(pred):
            category = "wrong_unknown"
        elif gold_seen:
            category = "wrong_gold_seen"
        else:
            category = "wrong_gold_missing"
        categories[category] += 1

        examples.append(
            {
                "query_id": qid,
                "category": category,
                "status": status,
                "gold_answer": gold,
                "predicted_answer": pred,
                "gold_seen_in_tools": gold_seen,
                "pred_seen_in_tools": pred_seen,
                "tool_counts": dict(Counter(name for name, _, _ in iter_tool_results(submission.get("messages") or []))),
                "lexical_verify_count": lexical_verify_count,
                "model_verify_count": model_verify_count,
                "supported_model_verify": supported_model_verify,
                "missing_docid_verify_calls": missing_docid_calls,
            }
        )

    return {
        "eval_summary": eval_summary,
        "category_counts": dict(categories),
        "status_counts": dict(statuses),
        "tool_counts": dict(tool_counts),
        "lexical_verify_supported_likely_counts": dict(lexical_verdicts),
        "model_verdict_counts": dict(model_verdicts),
        "wrong_supported_by_model_verifier": sum(
            1
            for example in examples
            if example["category"] != "correct" and example["supported_model_verify"] > 0
        ),
        "missing_docid_verify_queries": sum(
            1 for example in examples if example["missing_docid_verify_calls"] > 0
        ),
        "examples": examples,
    }


def print_section(title: str, rows: Iterable[Dict[str, Any]], limit: int) -> None:
    print(f"\n{title}")
    count = 0
    for row in rows:
        print(
            "{qid} | {category} | gold={gold} | pred={pred} | tools={tools} | model_supported={supported} | missing_docid_calls={missing}".format(
                qid=row["query_id"],
                category=row["category"],
                gold=compact(row["gold_answer"], 70),
                pred=compact(row["predicted_answer"], 90),
                tools=row["tool_counts"],
                supported=row["supported_model_verify"],
                missing=row["missing_docid_verify_calls"],
            )
        )
        count += 1
        if count >= limit:
            break


def main() -> None:
    args = parse_args()
    report = analyze_run(args.submission, args.eval, args.dataset)

    print("SUMMARY")
    print(json.dumps(report["eval_summary"], ensure_ascii=False))
    print("\nCOUNTS")
    for key in (
        "category_counts",
        "status_counts",
        "tool_counts",
        "lexical_verify_supported_likely_counts",
        "model_verdict_counts",
    ):
        print(f"{key}: {json.dumps(report[key], ensure_ascii=False, sort_keys=True)}")
    print(f"wrong_supported_by_model_verifier: {report['wrong_supported_by_model_verifier']}")
    print(f"missing_docid_verify_queries: {report['missing_docid_verify_queries']}")

    examples = report["examples"]
    print_section("WRONG BUT GOLD APPEARED IN TOOL OUTPUT", (row for row in examples if row["category"] == "wrong_gold_seen"), args.top)
    print_section("WRONG AND GOLD NOT FOUND IN TOOL OUTPUT", (row for row in examples if row["category"] == "wrong_gold_missing"), args.top)
    print_section(
        "WRONG WITH MODEL VERIFIER SUPPORTED",
        (row for row in examples if row["category"] != "correct" and row["supported_model_verify"] > 0),
        args.top,
    )
    print_section(
        "VERIFY CALLS WITH MISSING DOCIDS",
        (row for row in examples if row["missing_docid_verify_calls"] > 0),
        args.top,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()
