"""SQLite access: users, documents, chunks and audit records."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at REAL NOT NULL,
  token_epoch INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, display_name TEXT NOT NULL, stored_name TEXT NOT NULL,
  sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, pages INTEGER NOT NULL DEFAULT 0,
  chunk_count INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL, error TEXT, extraction_method TEXT,
  created_at REAL NOT NULL, updated_at REAL NOT NULL,
  warnings TEXT, ocr_pages INTEGER NOT NULL DEFAULT 0,
  UNIQUE(owner_id, sha256));
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, owner_id TEXT NOT NULL, page INTEGER NOT NULL,
  idx INTEGER NOT NULL, section TEXT, text TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'text');
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, user_id TEXT, event TEXT NOT NULL,
  doc_id TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit(user_id, ts);
CREATE INDEX IF NOT EXISTS idx_docs_owner ON documents(owner_id);
"""

# Columns added after the first release: applied to databases created by older versions (data is preserved).
MIGRATIONS = [
    ("users", "token_epoch", "INTEGER NOT NULL DEFAULT 0"),
    ("documents", "warnings", "TEXT"),
    ("documents", "ocr_pages", "INTEGER NOT NULL DEFAULT 0"),
    ("chunks", "kind", "TEXT NOT NULL DEFAULT 'text'"),
]


def new_id() -> str:
    return uuid.uuid4().hex


class Database:
    def __init__(self, path: Path | str):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        for table, column, ddl in MIGRATIONS:
            cols = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if column not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, tuple(params))
            self._conn.commit()

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        with self._lock:
            row = self._conn.execute(sql, tuple(params)).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def delete_document_rows(self, doc_id: str, owner_id: str) -> None:
        """Remove a document's chunks and metadata in one transaction."""
        with self._lock:
            try:
                self._conn.execute("DELETE FROM chunks WHERE doc_id=? AND owner_id=?", (doc_id, owner_id))
                self._conn.execute("DELETE FROM documents WHERE id=? AND owner_id=?", (doc_id, owner_id))
                self._conn.commit()
            except sqlite3.Error:
                self._conn.rollback()
                raise

    def replace_chunks(self, doc_id: str, owner_id: str, rows: list[tuple]) -> None:
        """Insert all chunks of a document atomically (rows: id, page, idx, section, text, kind)."""
        with self._lock:
            try:
                self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
                self._conn.executemany(
                    "INSERT INTO chunks(id, doc_id, owner_id, page, idx, section, text, kind) VALUES (?,?,?,?,?,?,?,?)",
                    [(r[0], doc_id, owner_id, r[1], r[2], r[3], r[4], r[5]) for r in rows],
                )
                self._conn.commit()
            except sqlite3.Error:
                self._conn.rollback()
                raise

    def ping(self) -> bool:
        try:
            self.one("SELECT 1 AS ok")
            return True
        except sqlite3.Error:
            return False

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- audit ----
    def audit(self, event: str, user_id: str | None = None, doc_id: str | None = None, **detail: Any) -> None:
        """Store an audit event. Never pass document text or secrets in `detail`."""
        self.execute(
            "INSERT INTO audit(ts, user_id, event, doc_id, detail) VALUES (?,?,?,?,?)",
            (time.time(), user_id, event, doc_id, json.dumps(detail, default=str)),
        )

    # ---- documents ----
    def mark_interrupted_documents(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE documents SET status='failed', error='Processing was interrupted by a restart. Delete and re-upload.', "
                "updated_at=? WHERE status IN ('processing','pending')",
                (time.time(),),
            )
            self._conn.commit()
            return cur.rowcount
