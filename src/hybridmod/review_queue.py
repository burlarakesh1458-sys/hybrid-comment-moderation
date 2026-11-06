"""Stage 4: SQLite human-review queue for decisions that could lead to an account penalty."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_queue (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    comment_id    TEXT,
    text          TEXT NOT NULL,
    proposed      TEXT NOT NULL,          -- automated action (hide / allow)
    stage         TEXT NOT NULL,
    category      TEXT,
    p_toxic       REAL,
    review_reason TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',   -- pending | resolved
    human_action  TEXT,                   -- allow | hide | penalize
    reviewer      TEXT,
    note          TEXT,
    created_at    REAL NOT NULL,
    resolved_at   REAL
);
CREATE INDEX IF NOT EXISTS ix_status ON review_queue(status, created_at);
"""

HUMAN_ACTIONS = {"allow", "hide", "penalize"}


class ReviewQueue:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def enqueue(self, comment_id: str | None, text: str, decision) -> int:
        d = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO review_queue (comment_id, text, proposed, stage, category, p_toxic,"
                " review_reason, decision_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (comment_id, text, d["action"], d["stage"], d.get("category"), d.get("p_toxic"),
                 d.get("review_reason") or "unspecified", json.dumps(d), time.time()),
            )
            return int(cur.lastrowid)

    def list(self, status: str = "pending", limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM review_queue WHERE status=? ORDER BY created_at, id LIMIT ?",
                (status, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get(self, item_id: int) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM review_queue WHERE id=?", (item_id,)).fetchone()
        return dict(r) if r else None

    def resolve(self, item_id: int, human_action: str, reviewer: str, note: str = "") -> dict:
        if human_action not in HUMAN_ACTIONS:
            raise ValueError(f"human_action must be one of {sorted(HUMAN_ACTIONS)}")
        item = self.get(item_id)
        if item is None:
            raise KeyError(item_id)
        if item["status"] != "pending":
            raise ValueError("item already resolved")
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE review_queue SET status='resolved', human_action=?, reviewer=?, note=?,"
                " resolved_at=? WHERE id=?",
                (human_action, reviewer, note, time.time(), item_id),
            )
        return self.get(item_id)  # type: ignore[return-value]

    def stats(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, review_reason, COUNT(*) n FROM review_queue GROUP BY 1,2"
            ).fetchall()
        out: dict = {"pending": 0, "resolved": 0, "by_reason": {}}
        for r in rows:
            out[r["status"]] += r["n"]
            out["by_reason"][r["review_reason"]] = out["by_reason"].get(r["review_reason"], 0) + r["n"]
        return out
