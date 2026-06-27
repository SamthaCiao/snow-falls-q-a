"""状态抽取：探索坐标 + 轨迹摘要（归纳主线，非问句清单）。"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from observation.config import ObserverConfig
from observation.character_index import get_known_entity_names
from observation.sanitize import (
    detect_prompt_example_leak,
    enrich_characters_from_turns,
    merge_user_interest_from_previous,
    sanitize_observation_state,
)
from observation.schema import (
    ExplorationCoordinates,
    SessionMeta,
    SessionObservationState,
    validate_no_identity_fields,
)

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = """你是文学对话的「阅读轨迹观测器」。只根据用户提供的对话内容输出 JSON。

## 两块内容必须互补、禁止重复（极重要）

### A. visible_summary ——「阅读旅程」（叙述式）
- 2-3 句，像回顾一段路：用户**先**问了什么方向、**后来**转向哪里
- 用「从…到…」「起初…随后…」等时间/路径语气
- 只讲故事线迁移，不写 bullet 关键词，不重复下方坐标字段

### B. exploration_coordinates ——「当前坐标」（结构化快照）
- 短字段：此刻聚焦什么、用什么角度、主题词、有无漂移、换方向建议
- 不写长段落，不 narrate 旅程

## JSON 结构（<> 内须从对话提炼，禁止照抄说明）
{
  "visible_summary": "<阅读旅程：叙述式 2-3 句>",
  "exploration_coordinates": {
    "topic_focus": "<≤10字>",
    "analysis_angle": "<分析角度>",
    "main_themes": ["<主题词>"],
    "recent_change": "<漂移描述或「稳定」>",
    "pivot_suggestion": "<一条新视角建议>"
  },
  "user_interest": { ... },
  "retrieval_hints": [...]
}

## 硬性规则
- 所有字段必须来自对话
- visible_summary 与 topic_focus/main_themes 文字不得大段重合
- 禁止「N轮/N个问题」统计句、字词碎片、人格标签
- user_interest.characters 只填作品内真实人名"""


class StateExtractor:
    def __init__(self, config: ObserverConfig):
        self.config = config
        load_dotenv("SUPER_MIND_API_KEY.env")
        load_dotenv(".env")
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")
        if bool(os.getenv("DEEPSEEK_API_KEY")):
            api_base = api_base or "https://api.deepseek.com"
        self._client: Optional[OpenAI] = None
        if api_key and config.use_llm_extractor:
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if api_base:
                kwargs["base_url"] = api_base
            self._client = OpenAI(**kwargs)
        default_model = os.getenv("OBSERVER_EXTRACTOR_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.model = config.extractor_model or default_model

    def extract(
        self,
        *,
        session_id: str,
        user_id: str,
        turns: List[Dict[str, Any]],
        previous: Optional[SessionObservationState],
        window_start_turn: int,
        window_end_turn: int,
        force_full: bool = False,
    ) -> SessionObservationState:
        user_messages = [
            (r.get("content") or "").strip()
            for r in turns
            if r.get("role") == "user" and (r.get("content") or "").strip()
        ]
        delta = self._format_turns(turns)

        if not self._client:
            logger.error("[用户观测][抽取] 未配置 LLM API")
            return self._failure_state(session_id, user_id, previous, window_start_turn, window_end_turn)

        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                merged = self._extract_with_llm(
                    delta,
                    None if force_full else previous,
                    user_messages=user_messages,
                    strict=(attempt > 1),
                )
                merged.session_meta = SessionMeta(
                    session_id=session_id,
                    user_id=user_id,
                    window_start_turn=window_start_turn,
                    window_end_turn=window_end_turn,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    extractor_version="llm-4.1",
                )
                merged = sanitize_observation_state(merged, user_messages=user_messages)
                if force_full and previous:
                    merged = merge_user_interest_from_previous(merged, previous)
                if not merged.user_interest.characters:
                    enrich_characters_from_turns(merged, user_messages)
                if detect_prompt_example_leak(merged):
                    raise ValueError("输出疑似照搬 prompt 示例，重试")
                exp = merged.user_interest.exploration
                logger.info(
                    "[用户观测][抽取] 成功 focus=%s themes=%s",
                    exp.topic_focus,
                    exp.main_themes,
                )
                return merged
            except Exception as e:
                last_error = e
                logger.warning("[用户观测][抽取] attempt=%d: %s", attempt, e)

        logger.error("[用户观测][抽取] 全部失败: %s", last_error)
        if previous:
            return previous
        return self._failure_state(session_id, user_id, None, window_start_turn, window_end_turn)

    @staticmethod
    def _failure_state(
        session_id: str,
        user_id: str,
        previous: Optional[SessionObservationState],
        window_start_turn: int,
        window_end_turn: int,
    ) -> SessionObservationState:
        if previous:
            return previous
        st = SessionObservationState.empty(session_id, user_id)
        st.visible_summary = "阅读轨迹暂时未能生成，请稍后点击「重新生成轨迹」重试。"
        st.session_meta.window_start_turn = window_start_turn
        st.session_meta.window_end_turn = window_end_turn
        st.session_meta.extractor_version = "failed"
        return st

    @staticmethod
    def _format_turns(turns: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        current_turn = None
        for row in turns:
            t = row.get("turn_index")
            if t != current_turn:
                current_turn = t
                lines.append(f"\n--- 第 {t} 轮 ---")
            role = "用户" if row.get("role") == "user" else "助手"
            content = (row.get("content") or "")[:600]
            lines.append(f"{role}: {content}")
        return "\n".join(lines).strip()

    def _extract_with_llm(
        self,
        delta_text: str,
        previous: Optional[SessionObservationState],
        *,
        user_messages: List[str],
        strict: bool = False,
    ) -> SessionObservationState:
        prev_json = previous.to_storage_dict() if previous else {}
        prev_json.pop("session_meta", None)

        questions_block = "\n".join(f"- {q[:120]}" for q in user_messages[-15:])
        strict_note = (
            "\n\n【重试】上次输出疑似模板/example 照抄。务必只根据下方「用户实际提问」和对话窗口提炼，"
            "不得出现 prompt 说明里的占位措辞。"
            if strict
            else ""
        )

        user_prompt = f"""已有阅读轨迹（JSON，合并更新；全量重算时覆盖旧摘要）：
{json.dumps(prev_json, ensure_ascii=False)[:2400]}

用户实际提问（探索坐标必须与此对应，不得脱离）：
{questions_block or "（暂无）"}

对话窗口：
{delta_text}

请输出完整 JSON。
- visible_summary：只写「阅读旅程」叙述（从A到B的路径），不要列关键词
- exploration_coordinates：只写当前状态快照，不要重复摘要里的句子{strict_note}"""

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.35 if strict else 0.25,
            "max_tokens": 1400,
        }
        try:
            kwargs["response_format"] = {"type": "json_object"}
            response = self._client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
        except Exception as e:
            logger.info("[用户观测][抽取] json_object 回退: %s", e)
            kwargs.pop("response_format", None)
            response = self._client.chat.completions.create(**kwargs)  # type: ignore[union-attr]

        raw = response.choices[0].message.content or "{}"
        data = validate_no_identity_fields(_parse_json_content(raw))
        if not data:
            raise ValueError(f"无法解析 JSON: {raw[:200]}")

        data = self._normalize_payload(data, previous)
        return SessionObservationState.from_storage_dict(data)

    @staticmethod
    def _normalize_payload(
        data: Dict[str, Any],
        previous: Optional[SessionObservationState],
    ) -> Dict[str, Any]:
        coords = data.get("exploration_coordinates") or {}
        if not isinstance(coords, dict):
            coords = {}

        ui_raw = data.get("user_interest") or {}
        if not isinstance(ui_raw, dict):
            ui_raw = {}

        themes = coords.get("main_themes") or ui_raw.get("themes") or []
        ui_raw["exploration"] = {
            "topic_focus": coords.get("topic_focus") or "",
            "analysis_angle": coords.get("analysis_angle") or "",
            "main_themes": themes if isinstance(themes, list) else [],
            "recent_change": coords.get("recent_change") or "稳定",
            "pivot_suggestion": coords.get("pivot_suggestion") or "",
        }
        data["user_interest"] = ui_raw

        exp = ExplorationCoordinates.model_validate(ui_raw["exploration"])
        data["exploration_coordinates_md"] = exp.to_markdown()

        if previous and not data.get("visible_summary"):
            data["visible_summary"] = previous.visible_summary

        if "session_meta" not in data:
            data["session_meta"] = {"session_id": "", "user_id": ""}
        if "retrieval_hints" not in data:
            data["retrieval_hints"] = []
        hints_raw = data.get("retrieval_hints") or []
        hints_norm: List[Dict[str, Any]] = []
        for h in hints_raw:
            if isinstance(h, str) and h.strip():
                hints_norm.append({"query": h.strip(), "source": "observation", "weight": 0.5})
            elif isinstance(h, dict) and h.get("query"):
                hints_norm.append(h)
        data["retrieval_hints"] = hints_norm

        return data


def _parse_json_content(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}
