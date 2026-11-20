"""On-disk LLM response cache and a cross-process spending ledger (SQLite).

The cache makes runs resumable and replays exact: the same request (model, effort, system
prompt, user message, schema, repeat index) always returns the stored response. Several
worker processes may share one database file.
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


class SqliteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT, "
                      "created REAL)")
            c.execute("CREATE TABLE IF NOT EXISTS spend (run_id TEXT, usd REAL, created REAL)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=60)


class ResponseCache(SqliteStore):
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


class BudgetExceeded(RuntimeError):
    pass


class Budget(SqliteStore):
    """Spending ledger for one run id, shared by all processes using the same file."""

    def __init__(self, path: str | Path, run_id: str, max_usd: float):
        super().__init__(path)
        self.run_id, self.max_usd = run_id, max_usd

    def spent(self) -> float:
        with self._conn() as c:
            row = c.execute("SELECT COALESCE(SUM(usd), 0) FROM spend WHERE run_id=?",
                            (self.run_id,)).fetchone()
        return float(row[0])

    def check(self) -> None:
        s = self.spent()
        if s >= self.max_usd:
            raise BudgetExceeded(f"run {self.run_id!r} spent ${s:.4f} of its ${self.max_usd:.2f} "
                                 "budget; raise --max-usd to continue (cached calls are free)")

    def add(self, usd: float) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO spend VALUES (?, ?, ?)", (self.run_id, float(usd), time.time()))
