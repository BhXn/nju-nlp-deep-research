from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .dataset_utils import load_jsonl
from .deep_research_agent import ResearchConfig, build_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-round BrowseComp-Plus Deep Research agent.")
    parser.add_argument("--dataset", default="browsecomp_plus_hard50.jsonl", help="Input JSONL dataset.")
    parser.add_argument("--output", default="runs/deep_research_submission.jsonl", help="Output submission JSONL.")
    parser.add_argument("--index-path", default="indexes/browsecomp_plus_bm25.sqlite", help="SQLite BM25 index path.")
    parser.add_argument("--model", default="qwen_auto", help="vLLM served model name.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="dummy", help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process.")
    parser.add_argument("--start", type=int, default=0, help="Start offset in the dataset.")
    parser.add_argument("--max-rounds", type=int, default=6, help="Maximum ReAct rounds after initial searches.")
    parser.add_argument("--max-initial-queries", type=int, default=7, help="Initial planned searches per question.")
    parser.add_argument("--top-k", type=int, default=8, help="BM25 results per search.")
    parser.add_argument("--auto-open-top-docs", type=int, default=0, help="Open top documents after initial search.")
    parser.add_argument("--max-tool-calls-per-round", type=int, default=4, help="Maximum tool calls accepted per ReAct round.")
    parser.add_argument("--max-total-tool-calls", type=int, default=36, help="Maximum total tool calls per query.")
    parser.add_argument("--max-no-new-info-rounds", type=int, default=2, help="Stop after this many no-new-info rounds.")
    parser.add_argument("--max-context-chars", type=int, default=24000, help="Compressed evidence context budget.")
    parser.add_argument("--snippet-max-chars", type=int, default=1000, help="Snippet character budget per search result.")
    parser.add_argument("--doc-max-chars", type=int, default=6000, help="Opened document character budget.")
    parser.add_argument("--max-evidence-docs", type=int, default=32, help="Maximum evidence documents in LLM context.")
    parser.add_argument("--planner-max-tokens", type=int, default=1024, help="Max tokens for planner calls.")
    parser.add_argument("--tool-max-tokens", type=int, default=1024, help="Max tokens for ReAct/tool-choice calls.")
    parser.add_argument("--answer-max-tokens", type=int, default=2048, help="Max tokens for answer synthesis calls.")
    parser.add_argument("--verifier-max-tokens", type=int, default=1024, help="Max tokens for verifier calls.")
    parser.add_argument("--verification-rounds", type=int, default=2, help="Answer verification and refinement rounds.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Agent model temperature.")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow Qwen thinking output. By default, chat_template_kwargs disables thinking for cleaner tool calls.",
    )
    parser.add_argument(
        "--no-model-planner",
        action="store_true",
        help="Use only deterministic decomposition for initial search planning.",
    )
    parser.add_argument(
        "--no-model-verifier",
        action="store_true",
        help="Skip LLM verification agent and keep lexical verify_claim records only.",
    )
    parser.add_argument(
        "--query-focused-snippet",
        action="store_true",
        help="Use query-centered search snippets instead of document-prefix snippets.",
    )
    parser.add_argument(
        "--prefer-heuristic-queries",
        action="store_true",
        help="Run deterministic query decomposition before model-planned queries.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ResearchConfig:
    return ResearchConfig(
        max_rounds=args.max_rounds,
        max_initial_queries=args.max_initial_queries,
        search_top_k=args.top_k,
        auto_open_top_docs=args.auto_open_top_docs,
        max_tool_calls_per_round=args.max_tool_calls_per_round,
        max_total_tool_calls=args.max_total_tool_calls,
        max_no_new_info_rounds=args.max_no_new_info_rounds,
        max_context_chars=args.max_context_chars,
        snippet_max_chars=args.snippet_max_chars,
        doc_max_chars=args.doc_max_chars,
        max_evidence_docs=args.max_evidence_docs,
        planner_max_tokens=args.planner_max_tokens,
        tool_max_tokens=args.tool_max_tokens,
        answer_max_tokens=args.answer_max_tokens,
        verifier_max_tokens=args.verifier_max_tokens,
        verification_rounds=args.verification_rounds,
        temperature=args.temperature,
        disable_thinking=not args.enable_thinking,
        use_model_planner=not args.no_model_planner,
        use_model_verifier=not args.no_model_verifier,
        query_focused_snippet=args.query_focused_snippet,
        prefer_heuristic_queries=args.prefer_heuristic_queries,
    )


def iter_rows(dataset_path: str, start: int = 0, limit: int | None = None) -> List[Dict[str, Any]]:
    rows = load_jsonl(dataset_path)
    if start:
        rows = rows[start:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def main() -> None:
    args = parse_args()
    config = build_config(args)
    agent = build_agent(
        index_path=args.index_path,
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        config=config,
    )

    rows = iter_rows(args.dataset, start=args.start, limit=args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as fout:
        for index, row in enumerate(rows, start=1):
            query_id = str(row.get("query_id", ""))
            print(f"[{index}/{len(rows)}] query_id={query_id}")
            record = agent.run(question=row["query"], query_id=query_id)
            records.append(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            print(
                "  status={status} answer={answer}".format(
                    status=record.get("status"),
                    answer=str(record.get("predicted_answer", ""))[:100],
                )
            )

    completed = sum(1 for record in records if record.get("status") == "completed")
    print(f"\nSaved {len(records)} records to {output_path}")
    print(f"Completed: {completed}/{len(records)}")


if __name__ == "__main__":
    main()
