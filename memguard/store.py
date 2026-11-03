"""SQLite 存储：记忆、原始轮次、召回日志、评估记录。

数据目录由调用方指定（AstrBot 下应为 data/plugin_data/<plugin_name>/），
绝不写回插件自身目录，避免更新插件时丢数据。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    text TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global',
    session TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    confidence REAL NOT NULL DEFAULT 0.6,
    importance REAL NOT NULL DEFAULT 0.5,
    hits INTEGER NOT NULL DEFAULT 0,
    misses INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'auto',
    created_at TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope, session);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    contamination REAL NOT NULL DEFAULT 0,
    labels TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrievals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL DEFAULT '',
    query TEXT NOT NULL,
    decision TEXT NOT NULL,
    hit_ids TEXT NOT NULL DEFAULT '[]',
    scores TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    metrics TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Memory:
    id: int
    key: str
    predicate: str
    object: str
    text: str
    scope: str
    session: str
    status: str
    confidence: float
    importance: float
    hits: int
    misses: int
    source: str
    created_at: str
    last_seen: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Memory":
        return cls(
            id=row["id"],
            key=row["key"],
            predicate=row["predicate"],
            object=row["object"],
            text=row["text"],
            scope=row["scope"],
            session=row["session"],
            status=row["status"],
            confidence=row["confidence"],
            importance=row["importance"],
            hits=row["hits"],
            misses=row["misses"],
            source=row["source"],
            created_at=row["created_at"],
            last_seen=row["last_seen"],
        )


class Store:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ memories

    def add_fact(
        self,
        *,
        predicate: str,
        obj: str,
        text: str,
        importance: float,
        scope: str,
        session: str,
        status: str = "active",
        source: str = "auto",
    ) -> tuple[int, str]:
        key = f"{predicate}:{obj}"
        now = utcnow()
        with self._lock:
            row = self._conn.execute("SELECT id, confidence FROM memories WHERE key = ?", (key,)).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE memories SET text=?, importance=MAX(importance, ?), last_seen=?, "
                    "confidence=MIN(1.0, confidence + 0.05) WHERE id=?",
                    (text, importance, now, row["id"]),
                )
                self._conn.commit()
                return int(row["id"]), "updated"

            cur = self._conn.execute(
                "INSERT INTO memories(key, predicate, object, text, scope, session, status, "
                "confidence, importance, source, created_at, last_seen) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, predicate, obj, text, scope, session, status, 0.6, importance, source, now, now),
            )
            self._conn.commit()
            return int(cur.lastrowid), "inserted"

    def active_memories(self, session: str | None = None) -> list[Memory]:
        with self._lock:
            if session is None:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE status='active' ORDER BY importance DESC, id DESC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE status='active' AND (scope='global' OR session=?) "
                    "ORDER BY importance DESC, id DESC",
                    (session,),
                ).fetchall()
        return [Memory.from_row(r) for r in rows]

    def all_memories(self, status: str | None = None, limit: int = 200) -> list[Memory]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE status=? ORDER BY id DESC LIMIT ?", (status, limit)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        return [Memory.from_row(r) for r in rows]

    def set_status(self, memory_id: int, status: str) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE memories SET status=? WHERE id=?", (status, memory_id))
            self._conn.commit()
            return cur.rowcount > 0

    def delete(self, memory_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def bump_hit(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET hits = hits + 1 WHERE id = ?", [(i,) for i in ids]
            )
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) c FROM memories GROUP BY status").fetchall()
            turns = self._conn.execute("SELECT COUNT(*) c FROM turns").fetchone()["c"]
            retrievals = self._conn.execute("SELECT COUNT(*) c FROM retrievals").fetchone()["c"]
            evals = self._conn.execute("SELECT COUNT(*) c FROM evals").fetchone()["c"]
        by_status = {r["status"]: r["c"] for r in rows}
        return {
            "active": by_status.get("active", 0),
            "quarantine": by_status.get("quarantine", 0),
            "archived": by_status.get("archived", 0),
            "turns": turns,
            "retrievals": retrievals,
            "evals": evals,
        }

    # ------------------------------------------------------------ turns / logs

    def log_turn(self, session: str, role: str, text: str, contamination: float, labels: list[str]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns(session, role, text, contamination, labels, created_at) VALUES(?,?,?,?,?,?)",
                (session, role, text[:2000], contamination, json.dumps(labels, ensure_ascii=False), utcnow()),
            )
            self._conn.commit()

    def log_retrieval(self, session: str, query: str, decision: str, hit_ids: list[int], scores: list[float]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO retrievals(session, query, decision, hit_ids, scores, created_at) VALUES(?,?,?,?,?,?)",
                (session, query[:500], decision, json.dumps(hit_ids), json.dumps([round(s, 4) for s in scores]), utcnow()),
            )
            self._conn.commit()

    def retrieval_report(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT decision, COUNT(*) c FROM retrievals GROUP BY decision"
            ).fetchall()
        return [{"decision": r["decision"], "count": r["c"]} for r in rows]

    # ------------------------------------------------------------ eval

    def save_eval(self, name: str, metrics: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO evals(name, metrics, created_at) VALUES(?,?,?)",
                (name, json.dumps(metrics, ensure_ascii=False), utcnow()),
            )
            self._conn.commit()

    def last_evals(self, limit: int = 5) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM evals ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [
            {"name": r["name"], "metrics": json.loads(r["metrics"]), "created_at": r["created_at"]}
            for r in rows
        ]
