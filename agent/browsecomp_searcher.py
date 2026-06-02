import argparse
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def snippetize(text: str, max_chars: Optional[int] = None) -> str:
    if not max_chars or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _resolve_existing_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if candidate.is_absolute():
        raise FileNotFoundError(f"Path does not exist: {candidate}")

    project_root = Path(__file__).resolve().parent.parent
    attempted = [candidate, project_root / candidate]
    for maybe in attempted[1:]:
        if maybe.exists():
            return maybe.resolve()
    attempted_str = ", ".join(str(p) for p in attempted)
    raise FileNotFoundError(f"Path does not exist: {candidate}. Attempted: {attempted_str}")


def _resolve_corpus_data_dir(corpus_path: str | Path) -> Path:
    corpus_path = _resolve_existing_path(corpus_path)
    if corpus_path.is_dir() and (corpus_path / "data").exists():
        data_dir = corpus_path / "data"
    else:
        data_dir = corpus_path
    if not data_dir.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {corpus_path}")
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")
    return data_dir


def iter_corpus_rows(corpus_path: str | Path, batch_size: int = 128) -> Iterable[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required to read browsecomp-plus-corpus parquet files. "
            "Please install it with `pip install pyarrow`."
        ) from exc

    data_dir = _resolve_corpus_data_dir(corpus_path)
    for parquet_path in sorted(data_dir.glob("*.parquet")):
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["docid", "text", "url"]):
            for row in batch.to_pylist():
                yield row


def build_sqlite_bm25_index(
    corpus_path: str | Path,
    index_path: str | Path,
    overwrite: bool = False,
    batch_size: int = 128,
) -> Dict[str, Any]:
    corpus_path = Path(corpus_path)
    index_path = Path(index_path)
    if index_path.exists():
        if not overwrite:
            raise FileExistsError(f"Index already exists: {index_path}")
        index_path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{index_path}{suffix}")
            if sidecar.exists():
                sidecar.unlink()

    index_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_path)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute("PRAGMA cache_size = -200000")
        connection.execute(
            """
            CREATE TABLE documents (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                docid TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                url TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE VIRTUAL TABLE documents_fts
            USING fts5(
                docid UNINDEXED,
                text,
                content='documents',
                content_rowid='rowid',
                tokenize='unicode61'
            )
            """
        )

        count = 0
        for row in iter_corpus_rows(corpus_path=corpus_path, batch_size=batch_size):
            docid = str(row.get("docid", "")).strip()
            text = str(row.get("text", "")).strip()
            url = str(row.get("url", "")).strip()
            if not docid or not text:
                continue
            cursor = connection.execute(
                "INSERT INTO documents(docid, text, url) VALUES (?, ?, ?)",
                (docid, text, url),
            )
            rowid = int(cursor.lastrowid)
            connection.execute(
                "INSERT INTO documents_fts(rowid, docid, text) VALUES (?, ?, ?)",
                (rowid, docid, text),
            )
            count += 1
            if count % max(batch_size * 10, 1000) == 0:
                connection.commit()

        connection.commit()
        connection.execute("INSERT INTO documents_fts(documents_fts) VALUES ('optimize')")
        connection.commit()
        return {"index_path": str(index_path), "num_documents": count}
    finally:
        connection.close()


class BrowseCompBM25Searcher:
    STOPWORDS = {
        "about",
        "after",
        "also",
        "among",
        "answer",
        "another",
        "because",
        "before",
        "being",
        "between",
        "certain",
        "could",
        "during",
        "first",
        "found",
        "from",
        "given",
        "have",
        "into",
        "last",
        "later",
        "more",
        "name",
        "noted",
        "other",
        "particular",
        "published",
        "question",
        "same",
        "some",
        "that",
        "their",
        "there",
        "these",
        "this",
        "those",
        "through",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
        "year",
    }

    @classmethod
    def parse_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--index-path", required=True, help="Path to the local SQLite BM25 index.")

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {self.index_path}. "
                "Please build it first with `python -m agent.build_bm25_index`."
            )
        self.connection = sqlite3.connect(f"file:{self.index_path}?mode=ro", uri=True, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row

    @property
    def search_type(self) -> str:
        return "bm25_sqlite_fts5"

    @staticmethod
    def _tokens_from_text(text: str) -> List[str]:
        pieces: List[str] = []
        seen = set()
        current: List[str] = []
        for ch in text.lower():
            if ch.isalnum() or ch == "_":
                current.append(ch)
            elif current:
                token = "".join(current)
                if token not in seen:
                    seen.add(token)
                    pieces.append(token)
                current = []
        if current:
            token = "".join(current)
            if token not in seen:
                pieces.append(token)
        return pieces

    @classmethod
    def _distinctive_tokens(cls, query: str, max_terms: int = 16) -> List[str]:
        tokens = []
        seen = set()
        for token in cls._tokens_from_text(query):
            if len(token) < 3 or token in cls.STOPWORDS:
                continue
            if token not in seen:
                seen.add(token)
                tokens.append(token)
        tokens.sort(key=lambda item: (0 if any(ch.isdigit() for ch in item) else 1, -len(item)))
        return tokens[:max_terms]

    @classmethod
    def _phrase_matches(cls, query: str, max_phrases: int = 4) -> List[str]:
        phrases: List[str] = []
        for match in re.finditer(r"[\"“”]([^\"“”]{4,80})[\"“”]", query):
            tokens = cls._tokens_from_text(match.group(1))
            if 1 < len(tokens) <= 8:
                phrases.append('"{}"'.format(" ".join(tokens)))
            if len(phrases) >= max_phrases:
                break
        return phrases

    @classmethod
    def _build_match_queries(cls, query: str) -> List[str]:
        phrases = cls._phrase_matches(query)
        distinctive = cls._distinctive_tokens(query, max_terms=18)
        all_tokens = cls._tokens_from_text(query)

        candidates: List[str] = []
        if phrases:
            candidates.append(" OR ".join(phrases))
        if len(distinctive) >= 4:
            candidates.append(" AND ".join(distinctive[: min(6, len(distinctive))]))
        if len(distinctive) >= 2:
            candidates.append(" OR ".join(distinctive[: min(14, len(distinctive))]))
        if all_tokens:
            candidates.append(" OR ".join(all_tokens[:32]))

        deduped: List[str] = []
        seen = set()
        for candidate in candidates:
            key = candidate.lower()
            if candidate and key not in seen:
                seen.add(key)
                deduped.append(candidate)
        return deduped

    @staticmethod
    def _tokenize_for_match(query: str) -> str:
        match_queries = BrowseCompBM25Searcher._build_match_queries(query)
        return match_queries[0] if match_queries else ""

    def _search_match_query(self, match_query: str, k: int) -> List[Dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT
                d.docid AS docid,
                d.text AS text,
                d.url AS url,
                bm25(documents_fts) AS raw_score
            FROM documents_fts
            JOIN documents d ON d.rowid = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY raw_score ASC
            LIMIT ?
            """,
            (match_query, int(k)),
        ).fetchall()

        return [
            {
                "docid": str(row["docid"]),
                "score": float(-row["raw_score"]),
                "text": str(row["text"]),
                "url": str(row["url"] or ""),
            }
            for row in rows
        ]

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        match_queries = self._build_match_queries(query)
        if not match_queries:
            return []

        merged: Dict[str, Dict[str, Any]] = {}
        for query_rank, match_query in enumerate(match_queries):
            try:
                results = self._search_match_query(match_query=match_query, k=max(k, 10))
            except sqlite3.OperationalError:
                continue
            for rank, item in enumerate(results):
                docid = item["docid"]
                rank_bonus = 1.0 / (1 + query_rank + rank)
                weighted_score = float(item["score"]) + rank_bonus
                existing = merged.get(docid)
                if existing is None or weighted_score > existing["_weighted_score"]:
                    enriched = dict(item)
                    enriched["_weighted_score"] = weighted_score
                    merged[docid] = enriched
            if len(merged) >= max(k * 3, 20):
                break

        ranked = sorted(merged.values(), key=lambda item: item["_weighted_score"], reverse=True)
        for item in ranked:
            item.pop("_weighted_score", None)
        return ranked[: int(k)]

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT docid, text, url FROM documents WHERE docid = ? LIMIT 1",
            (str(docid),),
        ).fetchone()
        if row is None:
            return None
        return {
            "docid": str(row["docid"]),
            "text": str(row["text"]),
            "url": str(row["url"] or ""),
        }
