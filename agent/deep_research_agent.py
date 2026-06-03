from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .browsecomp_searcher import BrowseCompBM25Searcher
from .tools import decompose_question_heuristic, get_deep_research_tool_specs_and_registry
from .vllm_client import VLLMClient


RESEARCH_SYSTEM_PROMPT = """You are a Deep Research Agent for BrowseComp-Plus.
You must answer using only the local BrowseComp-Plus tools and evidence returned by those tools.

Workflow:
1. Decompose the question into searchable clues.
2. Search with distinctive names, titles, dates, quoted phrases, and short clue combinations.
3. Open promising documents or find keywords inside them when snippets are not enough.
4. Keep track of confirmed facts, unresolved subgoals, and repeated searches.
5. Stop only when the answer is supported by evidence, when no new evidence is appearing, or when the round limit is reached.

Do not use Google, Bing, live web search, hidden gold answers, or outside knowledge.
Before finalizing, verify that the exact answer is supported by the retrieved evidence.

When you have enough evidence, answer exactly in this format:
Explanation: <brief evidence-based explanation>
Exact Answer: <short final answer only>
Confidence: <0-100>%

Do not include hidden reasoning, scratch work, or "Wait..." style analysis in the final answer."""


PLANNER_SYSTEM_PROMPT = """You are the planning agent in a multi-agent research system.
Your job is not to answer the question. Your job is to create search tasks for a local BM25 corpus.
Return strict JSON only, with this schema:
{
  "subquestions": ["..."],
  "search_queries": ["..."],
  "key_terms": ["..."],
  "must_verify": ["..."]
}

Good search queries are short and distinctive: names, titles, quoted phrases, dates, places, and rare clue combinations.
Avoid copying the whole question as one query. Prefer 2-6 term queries such as:
- person/title + year
- quoted phrase + place
- award/book/institution + author name
- rare clue term + date
Do not include a guessed final answer."""


ANSWER_SYSTEM_PROMPT = """You are the answer synthesis agent.
Use only the supplied research state and evidence. Prefer exact titles, names, dates, and quantities copied from evidence.
If evidence is incomplete, still provide the best-supported candidate and lower the confidence instead of refusing by default.
The Exact Answer line must contain only the answer string, not a sentence, hedge, explanation, or citation.
Do not include scratch work, alternatives, or hidden reasoning. If the evidence points to one candidate, output that candidate only on the Exact Answer line.
Return exactly:
Explanation: <brief evidence-based explanation>
Exact Answer: <short final answer only>
Confidence: <0-100>%"""


VERIFIER_SYSTEM_PROMPT = """You are the verification agent in a multi-agent research system.
Decide whether the candidate answer is supported by the supplied evidence.
Return strict JSON only:
{
  "verdict": "supported" | "unsupported" | "uncertain",
  "confidence": 0,
  "reason": "...",
  "missing_info": ["..."],
  "follow_up_queries": ["..."]
}

Use "supported" only when the evidence directly supports the exact answer."""


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[unused\d+\]", "", text)
    return text.strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _normalize_list(value: Any, limit: int) -> List[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value]
    elif isinstance(value, str):
        items = [piece.strip() for piece in re.split(r"[\n;]+", value)]
    else:
        items = []
    deduped: List[str] = []
    seen = set()
    for item in items:
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _extract_json_value(text: str) -> Optional[Any]:
    cleaned = _strip_thinking(text)
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")

    for start, char in enumerate(cleaned):
        if char not in "{[":
            continue
        opening = char
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for pos in range(start, len(cleaned)):
            current = cleaned[pos]
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == opening:
                depth += 1
            elif current == closing:
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : pos + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return None


def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            extracted = _extract_json_value(arguments)
            if isinstance(extracted, dict):
                return extracted
    return {}


def _clean_answer_string(answer: str) -> str:
    cleaned = _strip_thinking(str(answer)).strip().strip('"').strip()
    cleaned = re.sub(r"^\s*(?:Exact Answer|Final Answer|Answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    lowered = cleaned.lower()
    prefix_patterns = [
        r"^the (?:answer|name|club|company|person|title|date|number|language|location|university|species) is\s+",
        r"^the .*? is\s+",
        r"^it is\s+",
        r"^this is\s+",
    ]
    for pattern in prefix_patterns:
        match = re.match(pattern, lowered, flags=re.IGNORECASE)
        if match and len(cleaned) - match.end() >= 2:
            cleaned = cleaned[match.end() :].strip()
            break

    return cleaned.strip(" .")


def _extract_exact_answer(text: str) -> str:
    cleaned = _strip_thinking(text)
    json_value = _extract_json_value(cleaned)
    if isinstance(json_value, dict):
        for key in ("exact_answer", "final_answer", "answer"):
            if json_value.get(key):
                return _clean_answer_string(str(json_value[key]))

    for label in ("Exact Answer", "Final Answer", "Answer"):
        match = re.search(
            rf"{label}\s*:\s*(.+?)(?:\n(?:Confidence|Explanation|Reasoning|Notes?)\s*:|$)",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return _clean_answer_string(match.group(1))

    answer_like_patterns = [
        r"(?:the\s+(?:answer|name|club|company|person|title|date|number|language|location|university|species)\s+is|answer\s+is|it\s+is|would\s+be)\s+[\"“']?(.+?)[\"”']?(?:[.\n]|$)",
        r"(?:best\s+(?:answer|candidate)\s+is)\s+[\"“']?(.+?)[\"”']?(?:[.\n]|$)",
    ]
    for pattern in answer_like_patterns:
        matches = list(re.finditer(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL))
        for match in reversed(matches):
            candidate = _clean_answer_string(match.group(1))
            if 1 <= len(candidate) <= 160 and not re.search(
                r"\b(?:unknown|unable|insufficient|cannot determine|not enough evidence)\b",
                candidate,
                flags=re.IGNORECASE,
            ):
                return candidate

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    if lines[-1].lower().startswith("confidence:") and len(lines) >= 2:
        return _clean_answer_string(lines[-2])
    return _clean_answer_string(lines[-1])


def _ensure_final_format(candidate: str, fallback_confidence: int = 45) -> str:
    cleaned = _strip_thinking(candidate)
    if re.search(r"\bExact Answer\s*:", cleaned, flags=re.IGNORECASE):
        if not re.search(r"\bConfidence\s*:", cleaned, flags=re.IGNORECASE):
            cleaned = f"{cleaned.rstrip()}\nConfidence: {fallback_confidence}%"
        return cleaned
    exact_answer = _extract_exact_answer(cleaned)
    if not exact_answer:
        exact_answer = "Insufficient evidence"
    return (
        "Explanation: The answer is selected from the best available retrieved evidence.\n"
        f"Exact Answer: {exact_answer}\n"
        f"Confidence: {fallback_confidence}%"
    )


def _chat_extra_payload(model_name: str, disable_thinking: bool) -> Optional[Dict[str, Any]]:
    if disable_thinking:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


@dataclass
class ResearchConfig:
    max_rounds: int = 6
    max_initial_queries: int = 7
    search_top_k: int = 8
    auto_open_top_docs: int = 0
    max_tool_calls_per_round: int = 4
    max_total_tool_calls: int = 36
    max_no_new_info_rounds: int = 2
    snippet_max_chars: int = 1000
    doc_max_chars: int = 6000
    max_context_chars: int = 24000
    max_evidence_docs: int = 32
    planner_max_tokens: int = 1024
    tool_max_tokens: int = 1024
    answer_max_tokens: int = 2048
    verifier_max_tokens: int = 1024
    verification_rounds: int = 2
    temperature: float = 0.0
    disable_thinking: bool = True
    use_model_planner: bool = True
    use_model_verifier: bool = True
    query_focused_snippet: bool = False
    prefer_heuristic_queries: bool = False


@dataclass
class EvidenceStore:
    planned_subquestions: List[str] = field(default_factory=list)
    planned_queries: List[str] = field(default_factory=list)
    key_terms: List[str] = field(default_factory=list)
    must_verify: List[str] = field(default_factory=list)
    searches: List[Dict[str, Any]] = field(default_factory=list)
    documents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    seen_queries: set[str] = field(default_factory=set)
    verified_claims: List[Dict[str, Any]] = field(default_factory=list)
    no_new_info_rounds: int = 0

    def add_plan(self, plan: Dict[str, Any]) -> None:
        self.planned_subquestions = _normalize_list(plan.get("subquestions"), limit=12)
        self.planned_queries = _normalize_list(plan.get("search_queries"), limit=16)
        self.key_terms = _normalize_list(plan.get("key_terms"), limit=20)
        self.must_verify = _normalize_list(plan.get("must_verify"), limit=12)

    def add_search_results(self, query: str, results: Any) -> int:
        query_key = query.lower().strip()
        if query_key:
            self.seen_queries.add(query_key)

        new_docids: List[str] = []
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict) or "docid" not in item:
                    continue
                docid = str(item["docid"])
                if docid not in self.documents:
                    new_docids.append(docid)
                    self.documents[docid] = {
                        "docid": docid,
                        "score": item.get("score"),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                        "source_queries": [],
                        "windows": [],
                    }
                doc = self.documents[docid]
                if query and query not in doc["source_queries"]:
                    doc["source_queries"].append(query)
                if item.get("snippet") and len(str(item["snippet"])) > len(str(doc.get("snippet", ""))):
                    doc["snippet"] = item["snippet"]
                if item.get("score") is not None:
                    try:
                        doc["score"] = max(float(doc.get("score") or 0), float(item["score"]))
                    except (TypeError, ValueError):
                        doc["score"] = item["score"]

        self.searches.append(
            {
                "query": query,
                "num_results": len(results) if isinstance(results, list) else 0,
                "new_docids": new_docids,
            }
        )
        if new_docids:
            self.no_new_info_rounds = 0
        else:
            self.no_new_info_rounds += 1
        return len(new_docids)

    def add_tool_result(self, tool_name: str, arguments: Dict[str, Any], result: Any) -> int:
        if tool_name == "search":
            return self.add_search_results(str(arguments.get("query", "")), result)
        if tool_name in {"open_doc", "get_document"} and isinstance(result, dict):
            docid = str(result.get("docid", ""))
            if docid:
                doc = self.documents.setdefault(
                    docid,
                    {
                        "docid": docid,
                        "score": None,
                        "url": result.get("url", ""),
                        "snippet": "",
                        "source_queries": [],
                        "windows": [],
                    },
                )
                if result.get("text"):
                    doc["opened_text"] = result["text"]
                if result.get("url"):
                    doc["url"] = result["url"]
        elif tool_name == "find_in_doc" and isinstance(result, dict):
            docid = str(result.get("docid", ""))
            if docid:
                doc = self.documents.setdefault(
                    docid,
                    {
                        "docid": docid,
                        "score": None,
                        "url": result.get("url", ""),
                        "snippet": "",
                        "source_queries": [],
                        "windows": [],
                    },
                )
                for window in result.get("matches", []) or []:
                    if isinstance(window, dict) and window not in doc["windows"]:
                        doc["windows"].append(window)
        elif tool_name == "verify_claim" and isinstance(result, dict):
            self.verified_claims.append(result)
        return 0

    def top_docids(self, limit: int = 12) -> List[str]:
        return [doc["docid"] for doc in self._ranked_docs(limit)]

    def _ranked_docs(self, limit: int) -> List[Dict[str, Any]]:
        def sort_key(doc: Dict[str, Any]) -> Tuple[int, float]:
            try:
                score = float(doc.get("score") or 0)
            except (TypeError, ValueError):
                score = 0.0
            has_detail = int(bool(doc.get("opened_text") or doc.get("windows")))
            return (has_detail, score)

        return sorted(self.documents.values(), key=sort_key, reverse=True)[:limit]

    def to_context(self, max_chars: int, max_docs: int) -> str:
        parts: List[str] = []
        if self.planned_subquestions:
            parts.append("Subquestions:\n" + "\n".join(f"- {item}" for item in self.planned_subquestions[:8]))
        if self.must_verify:
            parts.append("Must verify:\n" + "\n".join(f"- {item}" for item in self.must_verify[:8]))
        if self.searches:
            search_lines = [
                f"- {entry['query']} | results={entry['num_results']} | new={len(entry['new_docids'])}"
                for entry in self.searches[-10:]
            ]
            parts.append("Recent searches:\n" + "\n".join(search_lines))

        doc_blocks: List[str] = []
        for rank, doc in enumerate(self._ranked_docs(max_docs), start=1):
            block_lines = [
                f"[Evidence {rank}] docid={doc['docid']} score={doc.get('score')} url={doc.get('url', '')}",
            ]
            if doc.get("source_queries"):
                block_lines.append("source_queries: " + "; ".join(doc["source_queries"][:3]))
            if doc.get("snippet"):
                block_lines.append("snippet:\n" + _truncate(str(doc["snippet"]), 1200))
            if doc.get("opened_text"):
                block_lines.append("opened_text:\n" + _truncate(str(doc["opened_text"]), 1800))
            for window in doc.get("windows", [])[:3]:
                if isinstance(window, dict) and window.get("text"):
                    matched = window.get("matched", "")
                    block_lines.append(f"keyword_window matched={matched}:\n{_truncate(str(window['text']), 1200)}")
            doc_blocks.append("\n".join(block_lines))
        if doc_blocks:
            parts.append("Evidence documents:\n" + "\n\n".join(doc_blocks))

        if self.verified_claims:
            verification_lines = [
                _truncate(_json_dumps(item), 1000)
                for item in self.verified_claims[-3:]
            ]
            parts.append("Verification history:\n" + "\n".join(verification_lines))

        context = "\n\n".join(parts)
        return _truncate(context, max_chars)

    def summary(self) -> Dict[str, Any]:
        return {
            "num_searches": len(self.searches),
            "num_documents": len(self.documents),
            "top_docids": self.top_docids(limit=20),
            "planned_queries": self.planned_queries,
            "key_terms": self.key_terms,
            "no_new_info_rounds": self.no_new_info_rounds,
            "verified_claims": self.verified_claims[-3:],
        }


class PlannerAgent:
    def __init__(self, client: VLLMClient, model_name: str, config: ResearchConfig) -> None:
        self.client = client
        self.model_name = model_name
        self.config = config

    def plan(self, question: str) -> Dict[str, Any]:
        fallback = decompose_question_heuristic(
            question,
            max_subquestions=max(4, self.config.max_initial_queries),
        )
        if not self.config.use_model_planner:
            return fallback

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        try:
            response = self.client.simple_chat(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=self.config.planner_max_tokens,
                extra_payload=_chat_extra_payload(self.model_name, self.config.disable_thinking),
            )
            content = response["choices"][0]["message"].get("content", "")
            parsed = _extract_json_value(content)
        except Exception:
            parsed = None

        if not isinstance(parsed, dict):
            return fallback

        model_queries = _normalize_list(parsed.get("search_queries"), limit=16)
        model_queries = [
            query
            for query in model_queries
            if len(query) <= 140 and len(query.split()) <= 12
        ]
        fallback_queries = fallback.get("search_queries", [])

        if self.config.prefer_heuristic_queries:
            search_queries = (
                fallback_queries[: self.config.max_initial_queries]
                + model_queries
                + fallback_queries[self.config.max_initial_queries :]
            )
        else:
            search_queries = model_queries + fallback_queries

        merged = {
            "subquestions": _normalize_list(parsed.get("subquestions"), limit=12)
            or fallback.get("subquestions", []),
            "search_queries": search_queries,
            "key_terms": _normalize_list(parsed.get("key_terms"), limit=20)
            + fallback.get("key_terms", []),
            "must_verify": _normalize_list(parsed.get("must_verify"), limit=12),
        }
        deduped_queries = _normalize_list(merged["search_queries"], limit=16)
        merged["search_queries"] = deduped_queries
        merged["key_terms"] = _normalize_list(merged["key_terms"], limit=20)
        return merged


class VerificationAgent:
    def __init__(self, client: VLLMClient, model_name: str, config: ResearchConfig) -> None:
        self.client = client
        self.model_name = model_name
        self.config = config

    def verify(self, question: str, candidate_answer: str, evidence_context: str) -> Dict[str, Any]:
        if not candidate_answer:
            return {
                "verdict": "unsupported",
                "confidence": 0,
                "reason": "No candidate answer was produced.",
                "missing_info": ["candidate answer"],
                "follow_up_queries": [],
            }
        if not self.config.use_model_verifier:
            return {
                "verdict": "uncertain",
                "confidence": 50,
                "reason": "Model verifier disabled.",
                "missing_info": [],
                "follow_up_queries": [],
            }

        user_message = (
            f"Question:\n{question}\n\n"
            f"Candidate answer:\n{candidate_answer}\n\n"
            f"Evidence:\n{_truncate(evidence_context, self.config.max_context_chars)}"
        )
        try:
            response = self.client.simple_chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=self.config.verifier_max_tokens,
                extra_payload=_chat_extra_payload(self.model_name, self.config.disable_thinking),
            )
            content = response["choices"][0]["message"].get("content", "")
            parsed = _extract_json_value(content)
        except Exception as exc:
            return {
                "verdict": "uncertain",
                "confidence": 0,
                "reason": f"Verifier call failed: {exc}",
                "missing_info": [],
                "follow_up_queries": [],
            }

        if not isinstance(parsed, dict):
            return {
                "verdict": "uncertain",
                "confidence": 0,
                "reason": "Verifier did not return parseable JSON.",
                "missing_info": [],
                "follow_up_queries": [],
            }

        verdict = str(parsed.get("verdict", "uncertain")).lower()
        if verdict not in {"supported", "unsupported", "uncertain"}:
            verdict = "uncertain"
        try:
            confidence = int(parsed.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0
        return {
            "verdict": verdict,
            "confidence": max(0, min(100, confidence)),
            "reason": str(parsed.get("reason", "")),
            "missing_info": _normalize_list(parsed.get("missing_info"), limit=8),
            "follow_up_queries": _normalize_list(parsed.get("follow_up_queries"), limit=5),
        }


class DeepResearchAgent:
    def __init__(
        self,
        searcher: BrowseCompBM25Searcher,
        client: VLLMClient,
        model_name: str,
        config: Optional[ResearchConfig] = None,
    ) -> None:
        self.searcher = searcher
        self.client = client
        self.model_name = model_name
        self.config = config or ResearchConfig()
        self.tool_specs, self.tool_registry = get_deep_research_tool_specs_and_registry(
            searcher=searcher,
            default_k=self.config.search_top_k,
            snippet_max_chars=self.config.snippet_max_chars,
            doc_max_chars=self.config.doc_max_chars,
            query_focused_snippet=self.config.query_focused_snippet,
        )
        self.planner = PlannerAgent(client=client, model_name=model_name, config=self.config)
        self.verifier = VerificationAgent(client=client, model_name=model_name, config=self.config)
        self._call_counter = 0

    def run(self, question: str, query_id: Optional[str] = None) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        state = EvidenceStore()
        status = "completed"
        draft_answer = ""

        try:
            self._run_initial_research(question=question, messages=messages, state=state)

            for round_id in range(1, self.config.max_rounds + 1):
                if self._count_tool_calls(messages) >= self.config.max_total_tool_calls:
                    status = "max_tool_calls_reached"
                    break
                if state.no_new_info_rounds >= self.config.max_no_new_info_rounds:
                    status = "no_new_information"
                    break

                round_answer = self._run_react_round(
                    question=question,
                    round_id=round_id,
                    messages=messages,
                    state=state,
                )
                if round_answer:
                    draft_answer = round_answer
                    break

            final_content = self._synthesize_answer(
                question=question,
                state=state,
                previous_answer=draft_answer,
            )

            final_content = self._verify_and_refine(
                question=question,
                candidate=final_content,
                messages=messages,
                state=state,
            )
            final_content = _ensure_final_format(final_content)
            messages.append({"role": "assistant", "content": final_content})
        except Exception as exc:
            status = "error"
            final_content = (
                "Explanation: The agent failed before completing the research loop.\n"
                f"Exact Answer: ERROR: {exc}\n"
                "Confidence: 0%"
            )
            messages.append({"role": "assistant", "content": final_content})

        predicted_answer = _extract_exact_answer(final_content)
        return {
            "query_id": str(query_id) if query_id is not None else None,
            "status": status,
            "predicted_answer": predicted_answer,
            "messages": messages,
            "state_summary": state.summary(),
            "current_subgoal": state.must_verify[0] if state.must_verify else "",
            "next_action_plan": "completed" if status == "completed" else status,
        }

    def _run_initial_research(
        self,
        question: str,
        messages: List[Dict[str, Any]],
        state: EvidenceStore,
    ) -> None:
        plan = self.planner.plan(question)
        state.add_plan(plan)
        self._record_manual_tool_call(
            messages=messages,
            state=state,
            tool_name="decompose_question",
            arguments={"question": question, "max_subquestions": self.config.max_initial_queries},
            result=plan,
            content="Planning initial research steps.",
        )

        queries = plan.get("search_queries", [])[: self.config.max_initial_queries]
        for query in queries:
            if not query:
                continue
            if query.lower().strip() in state.seen_queries:
                continue
            result = self.tool_registry["search"](query=query, top_k=self.config.search_top_k)
            self._record_manual_tool_call(
                messages=messages,
                state=state,
                tool_name="search",
                arguments={"query": query, "top_k": self.config.search_top_k},
                result=result,
                content="Executing planned search.",
            )

        if self.config.auto_open_top_docs > 0:
            for docid in state.top_docids(limit=self.config.auto_open_top_docs):
                result = self.tool_registry["open_doc"](docid=docid, max_chars=self.config.doc_max_chars)
                self._record_manual_tool_call(
                    messages=messages,
                    state=state,
                    tool_name="open_doc",
                    arguments={"docid": docid, "max_chars": self.config.doc_max_chars},
                    result=result,
                    content="Opening a high-ranked document for evidence.",
                )
        state.no_new_info_rounds = 0

    def _run_react_round(
        self,
        question: str,
        round_id: int,
        messages: List[Dict[str, Any]],
        state: EvidenceStore,
    ) -> str:
        call_messages = self._build_round_messages(question=question, round_id=round_id, state=state)
        response = self.client.simple_chat(
            model=self.model_name,
            messages=call_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.tool_max_tokens,
            tools=self.tool_specs,
            tool_choice="auto",
            extra_payload=_chat_extra_payload(self.model_name, self.config.disable_thinking),
        )
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or self._extract_tool_calls_from_content(content)

        assistant_message: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls[: self.config.max_tool_calls_per_round]
        messages.append(assistant_message)

        if not tool_calls:
            if content.strip():
                return content
            return ""

        for tool_call in tool_calls[: self.config.max_tool_calls_per_round]:
            tool_name, arguments, result = self._execute_tool_call(tool_call, state)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", self._next_call_id(tool_name)),
                    "content": _json_dumps(result),
                }
            )
            state.add_tool_result(tool_name, arguments, result)
        return ""

    def _build_round_messages(self, question: str, round_id: int, state: EvidenceStore) -> List[Dict[str, Any]]:
        evidence_context = state.to_context(
            max_chars=self.config.max_context_chars,
            max_docs=self.config.max_evidence_docs,
        )
        instruction = (
            f"Original question:\n{question}\n\n"
            f"Research round: {round_id}/{self.config.max_rounds}\n\n"
            f"Current compressed research state:\n{evidence_context}\n\n"
            "Decide the next action. If evidence is enough, provide the final answer in the required format. "
            "Otherwise call one or more tools. Avoid repeating previous searches unless you change the query."
        )
        return [
            {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

    def _synthesize_answer(self, question: str, state: EvidenceStore, previous_answer: str = "") -> str:
        evidence_context = state.to_context(
            max_chars=self.config.max_context_chars,
            max_docs=self.config.max_evidence_docs,
        )
        user_message = (
            f"Original question:\n{question}\n\n"
            f"Previous draft answer, if any:\n{previous_answer or '(none)'}\n\n"
            f"Compressed evidence state:\n{evidence_context}\n\n"
            "Now synthesize the best final answer."
        )
        try:
            response = self.client.simple_chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=self.config.answer_max_tokens,
                extra_payload=_chat_extra_payload(self.model_name, self.config.disable_thinking),
            )
            return response["choices"][0]["message"].get("content", "")
        except Exception:
            return previous_answer or (
                "Explanation: The evidence gathered so far is insufficient for a confident answer.\n"
                "Exact Answer: Insufficient evidence\n"
                "Confidence: 0%"
            )

    def _verify_and_refine(
        self,
        question: str,
        candidate: str,
        messages: List[Dict[str, Any]],
        state: EvidenceStore,
    ) -> str:
        current = candidate
        for _ in range(max(1, self.config.verification_rounds)):
            exact_answer = _extract_exact_answer(current)
            evidence_context = state.to_context(
                max_chars=self.config.max_context_chars,
                max_docs=self.config.max_evidence_docs,
            )
            verification = self.verifier.verify(
                question=question,
                candidate_answer=exact_answer,
                evidence_context=evidence_context,
            )
            self._record_manual_tool_call(
                messages=messages,
                state=state,
                tool_name="verify_claim",
                arguments={"claim": exact_answer, "docids": state.top_docids(limit=8)},
                result=verification,
                content="Verifying candidate answer against retrieved evidence.",
            )

            if verification["verdict"] == "supported":
                return current

            follow_up_queries = verification.get("follow_up_queries", [])
            new_searches = 0
            for query in follow_up_queries[:3]:
                if not query or query.lower().strip() in state.seen_queries:
                    continue
                result = self.tool_registry["search"](query=query, top_k=self.config.search_top_k)
                self._record_manual_tool_call(
                    messages=messages,
                    state=state,
                    tool_name="search",
                    arguments={"query": query, "top_k": self.config.search_top_k},
                    result=result,
                    content="Searching for missing verification evidence.",
                )
                new_searches += 1
            if new_searches == 0:
                return current

            current = self._synthesize_answer(
                question=question,
                state=state,
                previous_answer=current,
            )
        return current

    def _record_manual_tool_call(
        self,
        messages: List[Dict[str, Any]],
        state: EvidenceStore,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        content: str = "",
    ) -> None:
        call_id = self._next_call_id(tool_name)
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": _json_dumps(arguments),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": call_id, "content": _json_dumps(result)})
        state.add_tool_result(tool_name, arguments, result)

    def _execute_tool_call(self, tool_call: Dict[str, Any], state: EvidenceStore) -> Tuple[str, Dict[str, Any], Any]:
        function = tool_call.get("function") or {}
        tool_name = str(function.get("name") or "").strip()
        arguments = _parse_tool_arguments(function.get("arguments"))
        if tool_name == "get_document":
            tool_name = "open_doc"
        if tool_name not in self.tool_registry:
            return tool_name, arguments, {"error": f"unknown tool: {tool_name}"}
        if tool_name == "search":
            query = str(arguments.get("query", "")).strip()
            if query.lower() in state.seen_queries:
                return tool_name, arguments, {"query": query, "skipped": "duplicate_search", "results": []}
        try:
            result = self.tool_registry[tool_name](**arguments)
        except Exception as exc:
            result = {"error": str(exc), "tool": tool_name, "arguments": arguments}
        return tool_name, arguments, result

    def _extract_tool_calls_from_content(self, content: str) -> List[Dict[str, Any]]:
        parsed = _extract_json_value(content)
        if parsed is None:
            return []

        raw_calls: Iterable[Any]
        if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
            raw_calls = parsed["tool_calls"]
        elif isinstance(parsed, list):
            raw_calls = parsed
        else:
            raw_calls = [parsed]

        calls: List[Dict[str, Any]] = []
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            function = item.get("function") if isinstance(item.get("function"), dict) else item
            name = function.get("name") or function.get("tool") or function.get("tool_name")
            if not name:
                continue
            arguments = function.get("arguments") or {
                key: value
                for key, value in function.items()
                if key not in {"name", "tool", "tool_name", "function"}
            }
            calls.append(
                {
                    "id": item.get("id") or self._next_call_id(str(name)),
                    "type": "function",
                    "function": {
                        "name": str(name),
                        "arguments": _json_dumps(_parse_tool_arguments(arguments)),
                    },
                }
            )
        return calls

    def _next_call_id(self, tool_name: str) -> str:
        self._call_counter += 1
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", tool_name or "tool").strip("_") or "tool"
        return f"call_{self._call_counter}_{safe_name}"

    @staticmethod
    def _count_tool_calls(messages: List[Dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                total += len(message["tool_calls"])
        return total


def build_agent(
    index_path: str,
    model_name: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "dummy",
    config: Optional[ResearchConfig] = None,
) -> DeepResearchAgent:
    searcher = BrowseCompBM25Searcher(index_path=index_path)
    client = VLLMClient(base_url=base_url, api_key=api_key)
    return DeepResearchAgent(searcher=searcher, client=client, model_name=model_name, config=config)
