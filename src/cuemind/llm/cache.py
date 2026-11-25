"""On-disk LLM response cache (SQLite).

The cache makes runs resumable and replays exact: the same request (model, temperature,
system prompt, user message, schema, repeat index) always returns the stored response.
Several worker processes may share one database file. Failed calls are never cached.

Note: the default path `results/cache/llm_cache.sqlite` is relative to the current working
directory, so run scripts from the repository root.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def request_key(**parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT, "
                      "created REAL)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=60)

    def get(self, key: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT value FROM responses WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
                      (key, json.dumps(value, default=str), time.time()))

    def __len__(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
