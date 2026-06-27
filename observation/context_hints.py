"""检索 hint 路由：与当前问题竞争相关度，不注入人格标签。"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Set, Tuple

from observation.state_store import ObservationStateStore

logger = logging.getLogger(__name__)


def _token_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ta = set(a.replace("，", " ").replace("。", " ").split())
    tb = set(b.replace("，", " ").replace("。", " ").split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


class ContextHintRouter:
    """从 observation state 选取与当前问题相关的检索 hint。"""

    def __init__(self, store: Optional[ObservationStateStore] = None):
        self.store = store or ObservationStateStore()

    def select_hint_queries(
        self,
        *,
        session_id: Optional[str],
        user_id: Optional[str],
        current_query: str,
        max_hints: int = 3,
        min_overlap: float = 0.08,
    ) -> Tuple[List[str], float]:
        """返回 (hint 列表, 耗时秒)。"""
        t0 = time.perf_counter()
        if not session_id or not user_id:
            return [], 0.0
        state = self.store.load(session_id, user_id=user_id)
        load_ms = (time.perf_counter() - t0) * 1000
        if not state or not state.retrieval_hints:
            if load_ms > 1:
                logger.info(
                    "[用户观测][hint] session=%s 无可用 hint，读 state=%.1fms",
                    session_id,
                    load_ms,
                )
            return [], time.perf_counter() - t0

        scored: List[tuple[float, str]] = []
        for hint in state.retrieval_hints:
            overlap = _token_overlap(current_query, hint.query)
            score = overlap * 0.7 + hint.weight * 0.3
            if overlap >= min_overlap or hint.weight >= 0.45:
                scored.append((score, hint.query))

        scored.sort(key=lambda x: x[0], reverse=True)
        seen: Set[str] = set()
        result: List[str] = []
        for _, q in scored:
            q = q.strip()
            if not q or q in seen:
                continue
            seen.add(q)
            result.append(q)
            if len(result) >= max_hints:
                break

        elapsed = time.perf_counter() - t0
        if result:
            logger.info(
                "[用户观测][hint] session=%s 选中 %d 条 总耗时=%.1fms (读state=%.1fms): %s",
                session_id,
                len(result),
                elapsed * 1000,
                load_ms,
                result,
            )
        return result, elapsed

    @staticmethod
    def merge_into_multi_queries(
        base_queries: List[str],
        hint_queries: List[str],
        max_total: int = 5,
    ) -> List[str]:
        merged: List[str] = []
        seen: Set[str] = set()
        for q in base_queries + hint_queries:
            q = (q or "").strip()
            if not q or q in seen:
                continue
            seen.add(q)
            merged.append(q)
            if len(merged) >= max_total:
                break
        return merged
