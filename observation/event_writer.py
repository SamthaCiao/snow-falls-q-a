"""主对话进程：对话事件写入器。"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, Optional

from observation.db import get_database

logger = logging.getLogger(__name__)


class ConversationEventWriter:
    """非阻塞 append-only 写入；主对话路径仅调用此模块。"""

    def __init__(self):
        self._db = get_database()
        self._session_turns: Dict[str, int] = {}
        self._lock = threading.Lock()

    @staticmethod
    def new_session_id() -> str:
        return f"sess_{uuid.uuid4().hex[:16]}"

    def _next_turn(self, session_id: str) -> int:
        with self._lock:
            if session_id not in self._session_turns:
                self._session_turns[session_id] = self._db.get_max_turn_index(session_id)
            self._session_turns[session_id] += 1
            return self._session_turns[session_id]

    def record_turn(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        import time
        t0 = time.perf_counter()
        turn_index = self._next_turn(session_id)
        try:
            event_id = self._db.append_turn(
                session_id=session_id,
                user_id=user_id,
                turn_index=turn_index,
                user_content=user_message,
                assistant_content=assistant_message,
                metadata=metadata,
                conversation_id=conversation_id,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "[用户观测][事件写入] session=%s turn=%d 耗时=%.1fms chars=%d",
                session_id,
                turn_index,
                elapsed_ms,
                len(user_message) + len(assistant_message),
            )
            return turn_index
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.warning("[用户观测][事件写入] 失败 耗时=%.1fms: %s", elapsed_ms, e)
            return -1


_writer_singleton: Optional[ConversationEventWriter] = None
_writer_lock = threading.Lock()


def get_event_writer() -> ConversationEventWriter:
    global _writer_singleton
    with _writer_lock:
        if _writer_singleton is None:
            _writer_singleton = ConversationEventWriter()
        return _writer_singleton
