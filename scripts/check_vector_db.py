"""Exit 0 only when chroma_db exactly matches the current chunks and model."""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.vectordb.integrity import check_integrity

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
DB_PATH = BASE_DIR / "chroma_db"


def main() -> int:
    try:
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            chunks = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[INVALID] chunks.json을 읽을 수 없습니다: {exc}")
        return 1

    errors = check_integrity(DB_PATH, chunks)
    if errors:
        print("[INVALID] 벡터 DB 재구축이 필요합니다.")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"[OK] 벡터 DB 정합성 확인: {len(chunks)}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
