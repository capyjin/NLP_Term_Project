"""Lightweight ChromaDB build-manifest and integrity helpers."""

import hashlib
import json
import sqlite3
from pathlib import Path

from src.embedding.config import EMBEDDING_DIMENSION, MODEL_NAME

SCHEMA_VERSION = 1
MANIFEST_NAME = "build_manifest.json"


def chunk_ids_digest(chunk_ids: list[str]) -> str:
    payload = "\n".join(sorted(chunk_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_manifest(chunks: list[dict]) -> dict:
    chunk_ids = [chunk["id"] for chunk in chunks]
    return {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_count": len(chunk_ids),
        "chunk_ids_sha256": chunk_ids_digest(chunk_ids),
    }


def write_manifest(db_path: Path, chunks: list[dict]) -> None:
    manifest_path = db_path / MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(expected_manifest(chunks), f, ensure_ascii=False, indent=2)


def check_integrity(db_path: Path, chunks: list[dict]) -> list[str]:
    errors = []
    chunk_ids = [chunk["id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        errors.append("chunks.json에 중복 ID가 있습니다.")

    manifest_path = db_path / MANIFEST_NAME
    if not manifest_path.is_file():
        errors.append(f"{MANIFEST_NAME}이 없습니다.")
    else:
        try:
            with open(manifest_path, encoding="utf-8") as f:
                actual_manifest = json.load(f)
            expected = expected_manifest(chunks)
            for key, expected_value in expected.items():
                if actual_manifest.get(key) != expected_value:
                    errors.append(
                        f"manifest {key} 불일치: "
                        f"{actual_manifest.get(key)!r} != {expected_value!r}"
                    )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"manifest를 읽을 수 없습니다: {exc}")

    sqlite_path = db_path / "chroma.sqlite3"
    if not sqlite_path.is_file():
        errors.append("chroma.sqlite3가 없습니다.")
        return errors

    try:
        uri = f"{sqlite_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            indexed_ids = {
                row[0] for row in conn.execute("SELECT embedding_id FROM embeddings")
            }
        finally:
            conn.close()
        expected_ids = set(chunk_ids)
        if indexed_ids != expected_ids:
            errors.append(
                "색인 ID 불일치: "
                f"DB {len(indexed_ids)}건 / chunks.json {len(expected_ids)}건"
            )
    except sqlite3.Error as exc:
        errors.append(f"ChromaDB를 읽을 수 없습니다: {exc}")

    return errors
