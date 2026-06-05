from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .dataset_utils import load_jsonl
from .deep_research_agent import _chat_extra_payload, _extract_exact_answer, _extract_json_value, _strip_thinking
from .vllm_client import VLLMClient


FUSION_SYSTEM_PROMPT = """You are a strict evidence judge for BrowseComp-Plus.
You will receive one hard question, candidate answers from several legal local-agent runs, and evidence snippets retrieved from the local BrowseComp-Plus corpus.

Your job is to choose the best final answer using only the supplied evidence. Do not use outside knowledge.

Rules:
- First identify the exact answer type requested by the question.
- Break the question into constraints and check whether each candidate satisfies the same evidence chain.
- Penalize answers that are only clue values, related entities, titles from a different hop, or the wrong answer type.
- Treat previous run predictions and verifier verdicts as untrusted hints, not proof.
- Prefer a candidate supported by direct evidence over a frequent candidate.
- If every listed candidate is wrong but the evidence clearly contains a better answer, you may choose a short answer copied exactly from the evidence.
- If evidence is incomplete, choose the best-supported short answer rather than refusing by default.

Return strict JSON only:
{
  "answer_type": "...",
  "constraints": ["..."],
  "candidate_judgments": [
    {
      "answer": "...",
      "score": 0,
      "matched_constraints": ["..."],
      "missing_constraints": ["..."],
      "supporting_docids": ["..."],
      "contradictions": ["..."]
    }
  ],
  "final_answer": "...",
  "confidence": 0,
  "explanation": "..."
}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse several Deep Research submissions with an evidence judge.")
    parser.add_argument("--dataset", default="browsecomp_plus_hard50.jsonl", help="Dataset JSONL. Gold answers are not read.")
    parser.add_argument(
        "--submission",
        action="append",
        required=True,
        help="Submission JSONL. May be repeated. Optional label syntax: label=path.",
    )
    parser.add_argument("--output", default="runs/deep_research_submission_fused.jsonl", help="Fused submission JSONL.")
    parser.add_argument("--model", default="qwen_auto", help="vLLM served model name for the judge.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="dummy", help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--max-candidates", type=int, default=16, help="Candidate answer budget per question.")
    parser.add_argument("--max-evidence-chars", type=int, default=30000, help="Evidence context budget per question.")
    parser.add_argument("--max-search-results-per-query", type=int, default=3, help="Search results retained from each tool call.")
    parser.add_argument("--snippet-chars", type=int, default=700, help="Characters retained per evidence snippet/window.")
    parser.add_argument("--judge-max-tokens", type=int, default=1536, help="Max tokens for the judge response.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge model temperature.")
    parser.add_argument("--enable-thinking", action="store_true", help="Allow Qwen thinking output.")
    parser.add_argument("--start", type=int, default=0, help="Dataset start offset.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of examples.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the model; write fallback fused records for parser checks.")
    return parser.parse_args()


def compact(text: Any, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def normalize(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def query_id(row: Dict[str, Any]) -> str:
    return str(row.get("query_id") or row.get("id") or row.get("qid") or row.get("question_id") or "")


def load_submission(spec: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    if "=" in spec:
        label, path_text = spec.split("=", 1)
        label = label.strip() or Path(path_text).stem
    else:
        path_text = spec
        label = Path(path_text).stem
    rows = {query_id(row): row for row in load_jsonl(path_text)}
    return label, rows


def parse_tool_call_names(messages: Iterable[Dict[str, Any]]) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    calls: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            call_id = str(tool_call.get("id") or "")
            function = tool_call.get("function") or {}
            name = str(function.get("name") or "")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if call_id and name:
                calls[call_id] = (name, arguments)
    return calls


def decode_payload(payload: Any) -> Any:
    if not isinstance(payload, str):
        return payload
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload


def is_bad_candidate(answer: str) -> bool:
    normalized = normalize(answer)
    if not normalized:
        return True
    if normalized in {"unknown", "none", "null", "error", "insufficient evidence"}:
        return True
    if normalized.startswith("confidence") or normalized.startswith("explanation"):
        return True
    if any(phrase in normalized for phrase in ("not enough", "unable to determine", "not provided", "cannot determine")):
        return True
    return False


def add_candidate(candidates: Dict[str, Dict[str, Any]], answer: Any, source: str, reason: str, weight: int = 1) -> None:
    cleaned = _extract_exact_answer(str(answer))
    cleaned = compact(cleaned, 180).strip(" .")
    if is_bad_candidate(cleaned):
        return
    key = normalize(cleaned)
    if not key:
        return
    entry = candidates.setdefault(key, {"answer": cleaned, "sources": [], "count": 0, "score": 0})
    source_marker = {"source": source, "reason": reason}
    if source_marker in entry["sources"]:
        return
    entry["count"] += 1
    entry["score"] += max(1, weight)
    entry["sources"].append(source_marker)


def extract_title_from_snippet(snippet: str) -> str:
    match = re.search(r"(?:^|\n)title:\s*(.+)", snippet or "", flags=re.IGNORECASE)
    if not match:
        return ""
    title = match.group(1).strip()
    title = re.sub(r"\s+[-–]\s+(Wikipedia|YouTube|IMDb|Amazon.*)$", "", title, flags=re.IGNORECASE)
    return compact(title, 160)


def maybe_add_short_claim(candidates: Dict[str, Dict[str, Any]], claim: Any, source: str) -> None:
    text = compact(claim, 180)
    if len(text) <= 90 and not re.search(r"\b(the|this|that)\b.*\b(is|was|were|are)\b", text, re.IGNORECASE):
        add_candidate(candidates, text, source, "short verify claim", weight=3)


def extract_from_submission(
    label: str,
    record: Dict[str, Any],
    max_search_results_per_query: int,
    snippet_chars: int,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    candidates: Dict[str, Dict[str, Any]] = {}
    evidence_docs: Dict[str, Dict[str, Any]] = {}
    verifier_notes: List[str] = []

    predicted = record.get("predicted_answer")
    add_candidate(candidates, predicted, label, "run predicted answer", weight=6)

    for message in reversed(record.get("messages") or []):
        if message.get("role") == "assistant" and message.get("content"):
            content = str(message.get("content") or "")
            if re.search(r"\b(?:Exact Answer|Final Answer|Answer)\s*:", content, flags=re.IGNORECASE):
                add_candidate(candidates, content, label, "assistant exact answer", weight=5)
                break

    call_names = parse_tool_call_names(record.get("messages") or [])
    for message in record.get("messages") or []:
        if message.get("role") != "tool":
            continue
        tool_name, arguments = call_names.get(str(message.get("tool_call_id") or ""), ("unknown_tool", {}))
        payload = decode_payload(message.get("content"))

        if tool_name == "search" and isinstance(payload, list):
            query = str(arguments.get("query") or "")
            for result in payload[:max_search_results_per_query]:
                if not isinstance(result, dict):
                    continue
                docid = str(result.get("docid") or "")
                if not docid:
                    continue
                snippet = str(result.get("snippet") or "")
                doc = evidence_docs.setdefault(
                    docid,
                    {
                        "docid": docid,
                        "url": result.get("url", ""),
                        "score": result.get("score"),
                        "source_queries": [],
                        "snippets": [],
                        "sources": [],
                    },
                )
                if query and query not in doc["source_queries"]:
                    doc["source_queries"].append(query)
                if label not in doc["sources"]:
                    doc["sources"].append(label)
                if snippet:
                    doc["snippets"].append(compact(snippet, snippet_chars))
                    title = extract_title_from_snippet(snippet)
                    if title:
                        add_candidate(candidates, title, label, f"retrieved title docid={docid}", weight=1)

        elif tool_name in {"open_doc", "get_document"} and isinstance(payload, dict):
            docid = str(payload.get("docid") or "")
            if docid:
                doc = evidence_docs.setdefault(
                    docid,
                    {
                        "docid": docid,
                        "url": payload.get("url", ""),
                        "score": None,
                        "source_queries": [],
                        "snippets": [],
                        "sources": [],
                    },
                )
                if label not in doc["sources"]:
                    doc["sources"].append(label)
                if payload.get("text"):
                    doc["snippets"].append(compact(payload.get("text"), snippet_chars))

        elif tool_name == "find_in_doc" and isinstance(payload, dict):
            docid = str(payload.get("docid") or "")
            doc = evidence_docs.setdefault(
                docid,
                {
                    "docid": docid,
                    "url": payload.get("url", ""),
                    "score": None,
                    "source_queries": [],
                    "snippets": [],
                    "sources": [],
                },
            )
            if label not in doc["sources"]:
                doc["sources"].append(label)
            for window in payload.get("matches") or []:
                if isinstance(window, dict) and window.get("text"):
                    doc["snippets"].append(compact(window.get("text"), snippet_chars))

        elif tool_name == "verify_claim" and isinstance(payload, dict):
            claim = payload.get("claim") or arguments.get("claim")
            maybe_add_short_claim(candidates, claim, label)
            note_bits = [f"[{label}] claim={compact(claim, 160)}"]
            if "supported_likely" in payload:
                note_bits.append(f"lexical_supported={payload.get('supported_likely')}")
                note_bits.append(f"overlap={payload.get('best_token_overlap')}")
            if "verdict" in payload:
                note_bits.append(f"model_verdict={payload.get('verdict')}")
                note_bits.append(f"confidence={payload.get('confidence')}")
                if payload.get("reason"):
                    note_bits.append(f"reason={compact(payload.get('reason'), 260)}")
            verifier_notes.append(" | ".join(note_bits))

    ranked_docs = sorted(
        evidence_docs.values(),
        key=lambda doc: (
            len(doc.get("sources") or []),
            float(doc.get("score") or 0) if str(doc.get("score") or "").replace(".", "", 1).isdigit() else 0.0,
        ),
        reverse=True,
    )
    metadata = {
        "source_status": record.get("status"),
        "source_prediction": predicted,
        "num_messages": len(record.get("messages") or []),
    }
    return candidates, ranked_docs, {"metadata": metadata, "verifier_notes": verifier_notes}


def merge_candidates(candidate_maps: Iterable[Dict[str, Dict[str, Any]]], max_candidates: int) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for candidate_map in candidate_maps:
        for key, candidate in candidate_map.items():
            entry = merged.setdefault(key, {"answer": candidate["answer"], "sources": [], "count": 0, "score": 0})
            entry["count"] += int(candidate.get("count") or 1)
            entry["score"] += int(candidate.get("score") or candidate.get("count") or 1)
            entry["sources"].extend(candidate.get("sources") or [])

    ranked = sorted(
        merged.values(),
        key=lambda item: (item["score"], item["count"], -len(item["answer"]), item["answer"].lower()),
        reverse=True,
    )
    return ranked[:max_candidates]


def merge_evidence(evidence_lists: Iterable[List[Dict[str, Any]]], max_evidence_chars: int) -> str:
    docs: Dict[str, Dict[str, Any]] = {}
    for evidence_list in evidence_lists:
        for doc in evidence_list:
            docid = str(doc.get("docid") or "")
            if not docid:
                continue
            merged = docs.setdefault(
                docid,
                {
                    "docid": docid,
                    "url": doc.get("url", ""),
                    "score": doc.get("score"),
                    "source_queries": [],
                    "snippets": [],
                    "sources": [],
                },
            )
            for key in ("source_queries", "snippets", "sources"):
                for value in doc.get(key) or []:
                    if value and value not in merged[key]:
                        merged[key].append(value)
            if not merged.get("url") and doc.get("url"):
                merged["url"] = doc.get("url")

    ranked = sorted(
        docs.values(),
        key=lambda doc: (len(doc["sources"]), len(doc["snippets"])),
        reverse=True,
    )
    blocks: List[str] = []
    remaining = max_evidence_chars
    for index, doc in enumerate(ranked, start=1):
        snippets = doc["snippets"][:3]
        if not snippets:
            continue
        block = "\n".join(
            [
                f"[Evidence {index}] docid={doc['docid']} sources={','.join(doc['sources'][:5])} url={doc.get('url', '')}",
                "queries: " + "; ".join(doc["source_queries"][:5]),
                "snippets:\n" + "\n---\n".join(snippets),
            ]
        )
        if len(block) + 4 > remaining:
            block = compact(block, max(0, remaining - 4))
        if block:
            blocks.append(block)
            remaining -= len(block) + 4
        if remaining <= 0:
            break
    return "\n\n".join(blocks)


def format_candidates(candidates: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for index, candidate in enumerate(candidates, start=1):
        source_counts = Counter(item["source"] for item in candidate.get("sources") or [])
        sources = ", ".join(f"{source}:{count}" for source, count in source_counts.most_common())
        lines.append(
            f"{index}. {candidate['answer']} | score={candidate.get('score', candidate['count'])} "
            f"| votes={candidate['count']} | sources={sources}"
        )
    return "\n".join(lines) or "(no usable candidates)"


def build_user_message(question: str, candidates: List[Dict[str, Any]], evidence_context: str, verifier_notes: List[str]) -> str:
    verifier_section = "\n".join(verifier_notes[:20])
    return (
        f"Question:\n{question}\n\n"
        f"Candidate answers from prior legal runs:\n{format_candidates(candidates)}\n\n"
        "Prior verifier notes are untrusted hints only:\n"
        f"{verifier_section or '(none)'}\n\n"
        f"Retrieved evidence:\n{evidence_context or '(no evidence retained)'}\n\n"
        "Choose the final answer now. Return strict JSON only."
    )


def parse_judge_response(content: str) -> Dict[str, Any]:
    parsed = _extract_json_value(content)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def final_answer_from_judgment(judgment: Dict[str, Any], fallback: str) -> str:
    answer = judgment.get("final_answer") or judgment.get("answer") or fallback
    answer = _extract_exact_answer(str(answer))
    return compact(answer, 220).strip(" .") or fallback


def final_content(answer: str, judgment: Dict[str, Any], fallback_confidence: int = 50) -> str:
    explanation = compact(judgment.get("explanation") or "Selected by candidate fusion over prior run evidence.", 700)
    try:
        confidence = int(judgment.get("confidence", fallback_confidence))
    except (TypeError, ValueError):
        confidence = fallback_confidence
    confidence = max(0, min(100, confidence))
    return f"Explanation: {explanation}\nExact Answer: {answer}\nConfidence: {confidence}%"


def iter_dataset(dataset_path: str, start: int, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows = load_jsonl(dataset_path)
    if start:
        rows = rows[start:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def main() -> None:
    args = parse_args()
    source_runs = [load_submission(spec) for spec in args.submission]
    client = None if args.dry_run else VLLMClient(base_url=args.base_url, api_key=args.api_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = iter_dataset(args.dataset, start=args.start, limit=args.limit)
    with output_path.open("w", encoding="utf-8") as fout:
        for index, row in enumerate(rows, start=1):
            qid = query_id(row)
            question = row.get("query") or row.get("question") or ""
            candidate_maps: List[Dict[str, Dict[str, Any]]] = []
            evidence_lists: List[List[Dict[str, Any]]] = []
            verifier_notes: List[str] = []
            source_metadata: Dict[str, Any] = {}

            for label, records in source_runs:
                record = records.get(qid)
                if not record:
                    continue
                candidates, evidence, metadata = extract_from_submission(
                    label=label,
                    record=record,
                    max_search_results_per_query=args.max_search_results_per_query,
                    snippet_chars=args.snippet_chars,
                )
                candidate_maps.append(candidates)
                evidence_lists.append(evidence)
                verifier_notes.extend(metadata.get("verifier_notes") or [])
                source_metadata[label] = metadata.get("metadata")

            candidates = merge_candidates(candidate_maps, max_candidates=args.max_candidates)
            fallback = candidates[0]["answer"] if candidates else "Insufficient evidence"
            evidence_context = merge_evidence(evidence_lists, max_evidence_chars=args.max_evidence_chars)
            user_message = build_user_message(question, candidates, evidence_context, verifier_notes)

            if args.dry_run:
                judgment = {
                    "final_answer": fallback,
                    "confidence": 0,
                    "explanation": "Dry run fallback; judge model was not called.",
                }
                raw_judge = json.dumps(judgment, ensure_ascii=False)
            else:
                assert client is not None
                response = client.simple_chat(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": FUSION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=args.temperature,
                    max_tokens=args.judge_max_tokens,
                    extra_payload=_chat_extra_payload(args.model, not args.enable_thinking),
                )
                raw_judge = _strip_thinking(response["choices"][0]["message"].get("content", ""))
                judgment = parse_judge_response(raw_judge)

            answer = final_answer_from_judgment(judgment, fallback=fallback)
            content = final_content(answer, judgment)
            record = {
                "query_id": qid,
                "status": "fused_dry_run" if args.dry_run else "fused",
                "predicted_answer": answer,
                "messages": [
                    {"role": "system", "content": FUSION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": raw_judge},
                    {"role": "assistant", "content": content},
                ],
                "state_summary": {
                    "source_runs": list(source_metadata.keys()),
                    "source_metadata": source_metadata,
                    "candidates": candidates,
                    "judge": judgment,
                },
                "current_subgoal": "candidate_fusion",
                "next_action_plan": "completed",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{index}/{len(rows)}] query_id={qid} fused_answer={answer[:100]}")

    print(f"\nSaved fused submission to {output_path}")


if __name__ == "__main__":
    main()
