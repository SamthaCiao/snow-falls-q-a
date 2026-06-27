"""SQLite 持久化：对话事件、观测状态、用户与会话游标。"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from observation.config import get_db_path

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    email TEXT,
    display_name TEXT,
    auth_provider TEXT DEFAULT 'anonymous',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    conversation_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS conversation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, turn_index, role)
);

CREATE INDEX IF NOT EXISTS idx_events_session ON conversation_events(session_id, turn_index);

CREATE TABLE IF NOT EXISTS observation_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    window_start_turn INTEGER NOT NULL,
    window_end_turn INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_obs_session ON observation_states(session_id, version DESC);

CREATE TABLE IF NOT EXISTS observer_cursor (
    session_id TEXT PRIMARY KEY,
    last_processed_turn INTEGER NOT NULL DEFAULT 0,
    last_processed_event_id INTEGER NOT NULL DEFAULT 0,
    pending_char_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
"""


class ObservationDatabase:
    """线程安全的 SQLite 访问层。"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA)
                conn.commit()
            finally:
                conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def upsert_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        display_name: Optional[str] = None,
        auth_provider: str = "anonymous",
    ) -> None:
        now = self._now()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO users (user_id, email, display_name, auth_provider, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    email=COALESCE(excluded.email, users.email),
                    display_name=COALESCE(excluded.display_name, users.display_name),
                    auth_provider=excluded.auth_provider,
                    updated_at=excluded.updated_at
                """,
                (user_id, email, display_name, auth_provider, now, now),
            )

    def ensure_session(
        self,
        session_id: str,
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> None:
        now = self._now()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT session_id FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE chat_sessions SET updated_at = ?, conversation_id = COALESCE(?, conversation_id) WHERE session_id = ?",
                    (now, conversation_id, session_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO chat_sessions (session_id, user_id, conversation_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, user_id, conversation_id, now, now),
                )
            conn.execute(
                """
                INSERT INTO observer_cursor (session_id, last_processed_turn, last_processed_event_id, pending_char_count, updated_at)
                VALUES (?, 0, 0, 0, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, now),
            )

    def append_turn(
        self,
        session_id: str,
        user_id: str,
        turn_index: int,
        user_content: str,
        assistant_content: str,
        metadata: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
    ) -> int:
        """写入一轮对话（user + assistant），返回 assistant event id。"""
        self.upsert_user(user_id)
        self.ensure_session(session_id, user_id, conversation_id)
        now = self._now()
        meta = json.dumps(metadata or {}, ensure_ascii=False)
        char_delta = len(user_content) + len(assistant_content)

        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO conversation_events
                (session_id, user_id, turn_index, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, 'user', ?, ?, ?)
                """,
                (session_id, user_id, turn_index, user_content, meta, now),
            )
            cur = conn.execute(
                """
                INSERT OR REPLACE INTO conversation_events
                (session_id, user_id, turn_index, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, 'assistant', ?, ?, ?)
                """,
                (session_id, user_id, turn_index, assistant_content, meta, now),
            )
            assistant_event_id = cur.lastrowid or 0
            conn.execute(
                """
                UPDATE observer_cursor
                SET pending_char_count = pending_char_count + ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (char_delta, now, session_id),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            return int(assistant_event_id)

    def get_max_turn_index(self, session_id: str) -> int:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT MAX(turn_index) AS m FROM conversation_events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return int(row["m"] or 0)

    def get_cursor(self, session_id: str) -> Dict[str, int]:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT last_processed_turn, last_processed_event_id, pending_char_count FROM observer_cursor WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return {"last_processed_turn": 0, "last_processed_event_id": 0, "pending_char_count": 0}
            return {
                "last_processed_turn": int(row["last_processed_turn"]),
                "last_processed_event_id": int(row["last_processed_event_id"]),
                "pending_char_count": int(row["pending_char_count"]),
            }

    def get_turns_since(
        self,
        session_id: str,
        after_turn: int,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT turn_index, role, content, metadata_json, created_at
                FROM conversation_events
                WHERE session_id = ? AND turn_index > ?
                ORDER BY turn_index ASC, CASE role WHEN 'user' THEN 0 ELSE 1 END
                LIMIT ?
                """,
                (session_id, after_turn, limit * 2),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_turns(self, session_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """读取 session 全部对话轮次（用于手动全量重抽取）。"""
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT turn_index, role, content, metadata_json, created_at
                FROM conversation_events
                WHERE session_id = ?
                ORDER BY turn_index ASC, CASE role WHEN 'user' THEN 0 ELSE 1 END
                LIMIT ?
                """,
                (session_id, limit * 2),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_sessions_needing_extraction(
        self,
        every_n_turns: int,
        char_threshold: int,
    ) -> List[str]:
        """返回待抽取的 session_id 列表。"""
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT cs.session_id,
                       COALESCE(oc.last_processed_turn, 0) AS last_turn,
                       COALESCE(oc.pending_char_count, 0) AS pending_chars,
                       COALESCE(MAX(ce.turn_index), 0) AS max_turn
                FROM chat_sessions cs
                LEFT JOIN observer_cursor oc ON cs.session_id = oc.session_id
                LEFT JOIN conversation_events ce ON cs.session_id = ce.session_id
                GROUP BY cs.session_id
                """
            ).fetchall()

        result: List[str] = []
        for row in rows:
            max_turn = int(row["max_turn"] or 0)
            last_turn = int(row["last_turn"] or 0)
            pending_chars = int(row["pending_chars"] or 0)
            delta_turns = max_turn - last_turn
            if delta_turns >= every_n_turns or pending_chars >= char_threshold:
                if max_turn > last_turn:
                    result.append(str(row["session_id"]))
        return result

    def save_observation_state(
        self,
        session_id: str,
        user_id: str,
        state_json: str,
        window_start_turn: int,
        window_end_turn: int,
    ) -> int:
        now = self._now()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM observation_states WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            version = int(row["v"]) + 1
            conn.execute(
                """
                INSERT INTO observation_states
                (session_id, user_id, version, state_json, window_start_turn, window_end_turn, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, version, state_json, window_start_turn, window_end_turn, now),
            )
            conn.execute(
                """
                UPDATE observer_cursor
                SET last_processed_turn = ?,
                    pending_char_count = 0,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (window_end_turn, now, session_id),
            )
            return version

    def get_latest_observation(self, session_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self.transaction() as conn:
            if user_id:
                row = conn.execute(
                    """
                    SELECT state_json, version, window_start_turn, window_end_turn, created_at, user_id
                    FROM observation_states
                    WHERE session_id = ? AND user_id = ?
                    ORDER BY version DESC LIMIT 1
                    """,
                    (session_id, user_id),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT state_json, version, window_start_turn, window_end_turn, created_at, user_id
                    FROM observation_states
                    WHERE session_id = ?
                    ORDER BY version DESC LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
        if not row:
            return None
        data = json.loads(row["state_json"])
        data["_meta"] = {
            "version": row["version"],
            "window_start_turn": row["window_start_turn"],
            "window_end_turn": row["window_end_turn"],
            "created_at": row["created_at"],
            "user_id": row["user_id"],
        }
        return data

    def get_session_owner(self, session_id: str) -> Optional[str]:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT user_id FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return str(row["user_id"]) if row else None


_db_singleton: Optional[ObservationDatabase] = None
_db_lock = threading.Lock()


def get_database() -> ObservationDatabase:
    global _db_singleton
    with _db_lock:
        if _db_singleton is None:
            _db_singleton = ObservationDatabase()
        return _db_singleton
