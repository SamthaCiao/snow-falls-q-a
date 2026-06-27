"""用户观测独立进程：轮询待抽取会话并更新 state。"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from observation.config import ObserverConfig, get_runtime_data_dir
from observation.db import get_database
from observation.extractor import StateExtractor
from observation.state_store import ObservationStateStore

logger = logging.getLogger(__name__)


class UserObserverWorker:
    def __init__(self, config: Optional[ObserverConfig] = None):
        self.config = config or ObserverConfig.load()
        self.db = get_database()
        self.store = ObservationStateStore()
        self.extractor = StateExtractor(self.config)
        self._llm_calls = 0

    def run_once(self, session_id: Optional[str] = None, *, force_full: bool = False) -> int:
        """处理一批待抽取会话，返回处理数量。"""
        self._llm_calls = 0
        if session_id:
            targets = [session_id]
        else:
            targets = self.db.list_sessions_needing_extraction(
                self.config.trigger_every_n_turns,
                self.config.trigger_char_threshold,
            )

        processed = 0
        for sid in targets:
            if self._llm_calls >= self.config.max_llm_calls_per_run:
                logger.info("已达本轮 LLM 调用上限 %d，停止", self.config.max_llm_calls_per_run)
                break
            if self._process_session(sid, force_full=force_full):
                processed += 1
        return processed

    def _process_session(self, session_id: str, *, force_full: bool = False) -> bool:
        t_total = time.perf_counter()
        owner = self.db.get_session_owner(session_id)
        if not owner:
            return False

        if force_full:
            turns = self.db.get_all_turns(session_id, limit=200)
            if not turns:
                return False
            turn_indices = sorted({int(t["turn_index"]) for t in turns})
            window_start = turn_indices[0]
            window_end = turn_indices[-1]
            previous = self.store.load(session_id, user_id=owner)
            logger.info("[用户观测][全量重抽取] session=%s turns=%d-%d", session_id, window_start, window_end)
        else:
            cursor = self.db.get_cursor(session_id)
            last_turn = cursor["last_processed_turn"]
            max_turn = self.db.get_max_turn_index(session_id)
            if max_turn <= last_turn:
                return False
            turns = self.db.get_turns_since(
                session_id,
                after_turn=last_turn,
                limit=self.config.max_turns_per_extract,
            )
            if not turns:
                return False
            turn_indices = sorted({int(t["turn_index"]) for t in turns})
            window_start = turn_indices[0]
            window_end = turn_indices[-1]
            t_load = time.perf_counter()
            previous = self.store.load(session_id, user_id=owner)
            load_ms = (time.perf_counter() - t_load) * 1000
            logger.debug("[用户观测] 读 state=%.0fms", load_ms)

        t_db = time.perf_counter()
        db_read_ms = (time.perf_counter() - t_db) * 1000

        t_extract = time.perf_counter()
        state = self.extractor.extract(
            session_id=session_id,
            user_id=owner,
            turns=turns,
            previous=previous,
            window_start_turn=window_start,
            window_end_turn=window_end,
            force_full=force_full,
        )
        from observation.star_map_render import attach_star_map_render
        state = attach_star_map_render(state)
        extract_ms = (time.perf_counter() - t_extract) * 1000

        t_save = time.perf_counter()
        version = self.store.save(state, window_start_turn=window_start, window_end_turn=window_end)
        save_ms = (time.perf_counter() - t_save) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000
        self._llm_calls += 1
        logger.info(
            "[用户观测][完成] session=%s v=%d turns=%d-%d force=%s | "
            "总=%.0fms 读库=%.0fms 抽取=%.0fms 写state=%.0fms | %s",
            session_id,
            version,
            window_start,
            window_end,
            force_full,
            total_ms,
            db_read_ms,
            extract_ms,
            save_ms,
            (state.visible_summary or "")[:60],
        )
        return True

    def run_loop(self) -> None:
        logger.info(
            "用户观测模块启动 poll=%ds trigger_n=%d char_threshold=%d db=%s",
            self.config.poll_interval_seconds,
            self.config.trigger_every_n_turns,
            self.config.trigger_char_threshold,
            get_runtime_data_dir() / "observation.db",
        )
        while True:
            try:
                n = self.run_once()
                if n:
                    logger.info("本轮处理 %d 个会话", n)
            except Exception as e:
                logger.exception("观测轮询异常: %s", e)
            time.sleep(self.config.poll_interval_seconds)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="用户观测模块（独立进程）")
    parser.add_argument("--once", action="store_true", help="只运行一轮后退出")
    parser.add_argument("--session-id", type=str, default=None, help="仅处理指定 session")
    parser.add_argument("--force-full", action="store_true", help="全量重抽取（忽略游标）")
    parser.add_argument("--run-id", type=str, default=None, help="dev：等同 --once，便于脚本编排")
    parser.add_argument("--date", type=str, default=None, help="dev：预留按日期重跑（当前与 --once 相同）")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    worker = UserObserverWorker()
    if args.once or args.run_id or args.date or args.session_id:
        n = worker.run_once(session_id=args.session_id, force_full=args.force_full)
        logger.info("单次运行完成，处理 %d 个会话", n)
        return 0

    worker.run_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
