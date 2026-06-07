import argparse
import json
from pathlib import Path

from .browsecomp_searcher import build_sqlite_bm25_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local SQLite FTS5 BM25 index for BrowseComp-Plus corpus.")
    parser.add_argument("--corpus-path", required=True, help="Path to corpus root or data directory.")
    parser.add_argument("--index-path", required=True, help="Output SQLite index path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing index file.")
    parser.add_argument("--batch-size", type=int, default=128, help="Parquet read batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_sqlite_bm25_index(
        corpus_path=Path(args.corpus_path),
        index_path=Path(args.index_path),
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
