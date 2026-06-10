"""Regression tests for lightweight vector DB integrity checks."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.vectordb.integrity import check_integrity, expected_manifest


class VectorDbIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name)
        self.chunks = [{"id": "chunk-a"}, {"id": "chunk-b"}]

        conn = sqlite3.connect(self.db_path / "chroma.sqlite3")
        try:
            conn.execute("CREATE TABLE embeddings (embedding_id TEXT)")
            conn.executemany(
                "INSERT INTO embeddings (embedding_id) VALUES (?)",
                [("chunk-a",), ("chunk-b",)],
            )
            conn.commit()
        finally:
            conn.close()

        with open(self.db_path / "build_manifest.json", "w", encoding="utf-8") as f:
            json.dump(expected_manifest(self.chunks), f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_exact_match_is_valid(self):
        self.assertEqual(check_integrity(self.db_path, self.chunks), [])

    def test_missing_indexed_id_is_invalid(self):
        conn = sqlite3.connect(self.db_path / "chroma.sqlite3")
        try:
            conn.execute("DELETE FROM embeddings WHERE embedding_id = 'chunk-b'")
            conn.commit()
        finally:
            conn.close()

        errors = check_integrity(self.db_path, self.chunks)
        self.assertTrue(any("색인 ID 불일치" in error for error in errors))

    def test_changed_chunks_invalidate_manifest(self):
        changed_chunks = self.chunks + [{"id": "chunk-c"}]
        errors = check_integrity(self.db_path, changed_chunks)
        self.assertTrue(any("manifest chunk_count 불일치" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
