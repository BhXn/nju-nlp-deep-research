import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .browsecomp_searcher import BrowseCompBM25Searcher, snippetize


def build_searcher(index_path: str) -> BrowseCompBM25Searcher:
    return BrowseCompBM25Searcher(index_path=index_path)


def retrieve_once(
    searcher: BrowseCompBM25Searcher,
    query: str,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    docs = searcher.search(query, k=k)
    return [
        {
            "docid": doc["docid"],
            "score": doc["score"],
            "snippet": snippetize(doc["text"], snippet_max_chars),
            "url": doc.get("url", ""),
        }
        for doc in docs
    ]


def format_rag_context(results: List[Dict[str, Any]]) -> str:
    blocks = []
    for rank, item in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Document {rank}]",
                    f"docid: {item['docid']}",
                    f"score: {item['score']}",
                    f"url: {item.get('url', '')}",
                    item["snippet"],
                ]
            )
        )
    return "\n\n".join(blocks)


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _tokenize_text(text: str) -> List[str]:
    stopwords = {
        "about",
        "after",
        "also",
        "answer",
        "before",
        "between",
        "book",
        "certain",
        "could",
        "first",
        "from",
        "have",
        "into",
        "last",
        "name",
        "question",
        "same",
        "that",
        "their",
        "there",
        "this",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
    }
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", text.lower())
    return [token for token in tokens if len(token) >= 4 and token not in stopwords]


def _extract_distinctive_terms(question: str, limit: int = 12) -> List[str]:
    terms: List[str] = []

    for match in re.finditer(r"[\"“”]([^\"“”]{3,80})[\"“”]", question):
        terms.append(match.group(1).strip())

    for match in re.finditer(r"\b(?:1[5-9]\d{2}|20\d{2})s?\b", question):
        terms.append(match.group(0))

    capitalized = re.findall(
        r"\b[A-Z][A-Za-z'&.-]+(?:\s+[A-Z][A-Za-z'&.-]+){1,5}\b",
        question,
    )
    terms.extend(capitalized)

    for token in _tokenize_text(question):
        if token not in terms:
            terms.append(token)

    deduped: List[str] = []
    seen = set()
    for term in terms:
        normalized = re.sub(r"\s+", " ", str(term)).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
        if len(deduped) >= limit:
            break
    return deduped


def decompose_question_heuristic(question: str, max_subquestions: int = 8) -> Dict[str, Any]:
    """Build a lightweight, deterministic decomposition for tool use and fallback planning."""
    pieces = [
        piece.strip(" \n\t-:;")
        for piece in re.split(r"(?<=[.;?])\s+|\n+", question)
        if piece.strip()
    ]
    if not pieces:
        pieces = [question.strip()]

    distinctive_terms = _extract_distinctive_terms(question)
    subquestions = pieces[: max(1, max_subquestions)]
    search_queries: List[str] = []

    if question.strip():
        search_queries.append(question.strip())
    search_queries.extend(distinctive_terms[: max_subquestions])

    for piece in subquestions:
        terms = _extract_distinctive_terms(piece, limit=4)
        if terms:
            search_queries.append(" ".join(terms))
        else:
            search_queries.append(piece)

    deduped_queries: List[str] = []
    seen = set()
    for query in search_queries:
        query = re.sub(r"\s+", " ", query).strip()
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        deduped_queries.append(query)

    return {
        "subquestions": subquestions,
        "search_queries": deduped_queries[: max_subquestions + 4],
        "key_terms": distinctive_terms,
    }


def _keyword_windows(
    text: str,
    keyword: str,
    max_windows: int,
    window_chars: int,
) -> List[Dict[str, Any]]:
    if not text or not keyword:
        return []

    lowered_text = text.lower()
    candidates = [keyword] + _tokenize_text(keyword)
    windows: List[Dict[str, Any]] = []
    seen_spans = set()

    for candidate in candidates:
        needle = candidate.lower().strip()
        if not needle:
            continue
        start = 0
        while len(windows) < max_windows:
            index = lowered_text.find(needle, start)
            if index < 0:
                break
            left = max(0, index - window_chars // 2)
            right = min(len(text), index + len(needle) + window_chars // 2)
            span_key = (left, right)
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                windows.append(
                    {
                        "matched": text[index : index + len(needle)],
                        "start": left,
                        "end": right,
                        "text": text[left:right].strip(),
                    }
                )
            start = index + max(1, len(needle))
        if len(windows) >= max_windows:
            break

    return windows


def get_search_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str) -> List[Dict[str, Any]]:
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    f"Search the BrowseComp-Plus BM25 index and return top-{k} results "
                    "with docid, score, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    return tools, {"search": search}


def get_agent_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str) -> List[Dict[str, Any]]:
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

    def get_document(docid: str) -> Dict[str, Any]:
        doc = searcher.get_document(docid)
        if doc is None:
            return {"docid": docid, "error": "document not found"}
        return doc

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    f"Search the BrowseComp-Plus BM25 index and return top-{k} results "
                    "with docid, score, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_document",
                "description": "Retrieve a full document by its docid.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id"},
                    },
                    "required": ["docid"],
                },
            },
        },
    ]
    return tools, {"search": search, "get_document": get_document}


def get_deep_research_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    default_k: int = 8,
    max_k: int = 12,
    snippet_max_chars: int = 1000,
    doc_max_chars: int = 6000,
    window_chars: int = 900,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    """Return the richer tool set used by the multi-round Deep Research agent."""

    def search(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = _clamp_int(top_k, default=default_k, minimum=1, maximum=max_k)
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

    def open_doc(docid: str, max_chars: Optional[int] = None) -> Dict[str, Any]:
        doc = searcher.get_document(docid)
        if doc is None:
            return {"docid": str(docid), "error": "document not found"}
        limit = _clamp_int(max_chars, default=doc_max_chars, minimum=500, maximum=doc_max_chars)
        return {
            "docid": doc["docid"],
            "url": doc.get("url", ""),
            "text": snippetize(doc["text"], limit),
            "truncated": len(doc["text"]) > limit,
        }

    def find_in_doc(
        docid: str,
        keyword: str,
        max_windows: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        doc = searcher.get_document(docid)
        if doc is None:
            return {"docid": str(docid), "keyword": keyword, "error": "document not found", "matches": []}
        n_windows = _clamp_int(max_windows, default=4, minimum=1, maximum=8)
        win_chars = _clamp_int(window_size, default=window_chars, minimum=200, maximum=2000)
        matches = _keyword_windows(doc["text"], keyword, max_windows=n_windows, window_chars=win_chars)
        return {
            "docid": doc["docid"],
            "url": doc.get("url", ""),
            "keyword": keyword,
            "matches": matches,
            "match_count": len(matches),
        }

    def decompose_question(question: str, max_subquestions: Optional[int] = None) -> Dict[str, Any]:
        limit = _clamp_int(max_subquestions, default=8, minimum=2, maximum=12)
        return decompose_question_heuristic(question, max_subquestions=limit)

    def verify_claim(
        claim: str,
        docids: Optional[Any] = None,
        keywords: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if isinstance(docids, str):
            candidate_docids = [item.strip() for item in re.split(r"[,;\s]+", docids) if item.strip()]
        elif isinstance(docids, list):
            candidate_docids = [str(item).strip() for item in docids if str(item).strip()]
        else:
            candidate_docids = [item["docid"] for item in search(claim, top_k=5)]

        if isinstance(keywords, str):
            terms = [item.strip() for item in re.split(r"[,;]", keywords) if item.strip()]
        elif isinstance(keywords, list):
            terms = [str(item).strip() for item in keywords if str(item).strip()]
        else:
            terms = _extract_distinctive_terms(claim, limit=10)
        if not terms:
            terms = _tokenize_text(claim)[:10]

        evidence: List[Dict[str, Any]] = []
        best_overlap = 0.0
        claim_tokens = set(_tokenize_text(claim))
        for docid in candidate_docids[:8]:
            doc = searcher.get_document(docid)
            if doc is None:
                continue
            doc_text_lower = doc["text"].lower()
            hits = [term for term in terms if term.lower() in doc_text_lower]
            token_hits = [token for token in claim_tokens if token in doc_text_lower]
            overlap = len(token_hits) / max(1, len(claim_tokens))
            best_overlap = max(best_overlap, overlap)
            windows: List[Dict[str, Any]] = []
            for term in hits[:3]:
                windows.extend(_keyword_windows(doc["text"], term, max_windows=1, window_chars=window_chars))
            if hits or token_hits:
                evidence.append(
                    {
                        "docid": docid,
                        "url": doc.get("url", ""),
                        "matched_terms": hits[:8],
                        "token_overlap": round(overlap, 3),
                        "windows": windows[:3],
                    }
                )

        return {
            "claim": claim,
            "docids_checked": candidate_docids[:8],
            "supported_likely": bool(evidence and best_overlap >= 0.35),
            "best_token_overlap": round(best_overlap, 3),
            "evidence": evidence,
        }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "Search the BrowseComp-Plus BM25 index. Use distinctive quoted phrases, names, "
                    "dates, titles, or short clue combinations. Returns ranked snippets."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "top_k": {
                            "type": "integer",
                            "description": f"Number of results to return, 1-{max_k}.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "open_doc",
                "description": "Open a retrieved document by docid and return a truncated full-text view.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id."},
                        "max_chars": {
                            "type": "integer",
                            "description": f"Maximum characters to return, capped at {doc_max_chars}.",
                        },
                    },
                    "required": ["docid"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_in_doc",
                "description": "Find keyword windows inside a retrieved document.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id."},
                        "keyword": {"type": "string", "description": "Keyword or phrase to locate."},
                        "max_windows": {"type": "integer", "description": "Maximum windows to return."},
                        "window_size": {"type": "integer", "description": "Approximate characters per window."},
                    },
                    "required": ["docid", "keyword"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "decompose_question",
                "description": "Split a complex question into searchable clues and candidate search queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Original complex question."},
                        "max_subquestions": {"type": "integer", "description": "Maximum items to return."},
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verify_claim",
                "description": (
                    "Check whether a candidate claim is lexically supported by retrieved documents. "
                    "Use this before finalizing an exact answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "Candidate answer or claim to verify."},
                        "docids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional document ids to check.",
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional support terms to look for.",
                        },
                    },
                    "required": ["claim"],
                },
            },
        },
    ]

    return tools, {
        "search": search,
        "open_doc": open_doc,
        "get_document": open_doc,
        "find_in_doc": find_in_doc,
        "decompose_question": decompose_question,
        "verify_claim": verify_claim,
    }
