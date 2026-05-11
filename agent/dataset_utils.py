import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_jsonl(path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows
