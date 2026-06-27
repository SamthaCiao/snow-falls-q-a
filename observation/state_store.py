"""观测状态读写封装。"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from observation.db import get_database
from observation.schema import SessionObservationState, validate_no_identity_fields

logger = logging.getLogger(__name__)


class ObservationStateStore:
    def __init__(self):
        self._db = get_database()

    def load(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionObservationState]:
        raw = self._db.get_latest_observation(session_id, user_id=user_id)
        if not raw:
            return None
        raw.pop("_meta", None)
        raw = validate_no_identity_fields(raw)
        try:
            return SessionObservationState.from_storage_dict(raw)
        except Exception as e:
            logger.warning("解析 observation state 失败: %s", e)
            return None

    def save(
        self,
        state: SessionObservationState,
        window_start_turn: int,
        window_end_turn: int,
    ) -> int:
        payload = validate_no_identity_fields(state.to_storage_dict())
        return self._db.save_observation_state(
            session_id=state.session_meta.session_id,
            user_id=state.session_meta.user_id,
            state_json=json.dumps(payload, ensure_ascii=False),
            window_start_turn=window_start_turn,
            window_end_turn=window_end_turn,
        )

    def get_visible_summary(self, session_id: str, user_id: str) -> Optional[str]:
        state = self.load(session_id, user_id=user_id)
        if not state or not state.visible_summary:
            return None
        return state.visible_summary

    def get_retrieval_hints(self, session_id: str, user_id: str) -> List[str]:
        state = self.load(session_id, user_id=user_id)
        if not state:
            return []
        return [h.query for h in state.retrieval_hints if h.query.strip()]
