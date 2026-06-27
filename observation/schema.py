"""
SessionObservationState — 探索坐标 Markdown + 轨迹摘要 + 检索 hint

设计原则：observation ≠ identity；UI 展示探索坐标，不堆字段清单。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


FORBIDDEN_IDENTITY_KEYS = frozenset({
    "user_personality",
    "user_profile",
    "personality_traits",
    "preference_labels",
    "用户画像",
    "人格标签",
})


class ExplorationCoordinates(BaseModel):
    """当前探索坐标（结构化；UI 由 to_markdown 渲染）。"""

    topic_focus: str = Field("", description="话题聚焦，≤10字")
    analysis_angle: str = Field("", description="切入视角")
    main_themes: List[str] = Field(
        default_factory=list,
        description="多轮对话归纳的主线/主题关键词（共同点）",
    )
    recent_change: str = Field("稳定", description="近期变化；无漂移则写「稳定」")
    pivot_suggestion: str = Field("", description="换方向建议")

    def to_markdown(self) -> str:
        themes = "、".join(t for t in self.main_themes if t) or "—"
        change = (self.recent_change or "稳定").strip()
        pivot = (self.pivot_suggestion or "").strip()
        lines = [
            "## 📍 当前探索坐标",
            "",
            f"- **话题聚焦**：{self.topic_focus or '—'}",
            f"- **切入视角**：{self.analysis_angle or '—'}",
            f"- **主线主题**：{themes}",
            f"- **近期变化**：{change}",
            "",
        ]
        if pivot:
            lines.append(f"> 🔄 尝试换个方向？{pivot}")
        return "\n".join(lines)


class UserInterestState(BaseModel):
    """内部结构化字段：供星图点亮与 retrieval hint，不在 UI 逐条展示。"""

    exploration: ExplorationCoordinates = Field(default_factory=ExplorationCoordinates)
    characters: List[str] = Field(default_factory=list)
    imagery_discussed: List[str] = Field(default_factory=list)
    explored_chapters: List[str] = Field(default_factory=list)
    confirmed_facts: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)

    # 旧版字段兼容（读取后迁移，不写入 UI）
    focus: str = ""
    unresolved: List[str] = Field(default_factory=list)
    understanding_change: str = ""
    exploration_angle: str = ""


class RetrievalHint(BaseModel):
    query: str
    source: str = Field("observation", description="hint 来源")
    weight: float = Field(0.5, ge=0.0, le=1.0)


class StarMapRenderPayload(BaseModel):
    """P2+P3 预计算星图渲染数据（observer 写入，前端只读）。"""

    lit_node_ids: List[str] = Field(default_factory=list)
    lit_count: int = 0
    constellation_lines: List[Dict[str, Any]] = Field(default_factory=list)
    progress: float = Field(0.0, ge=0.0, le=1.0)
    mobius_arc_points: List[List[float]] = Field(default_factory=list)
    mobius_full_points: List[List[float]] = Field(default_factory=list)
    lit_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    computed_at: str = ""


class SessionMeta(BaseModel):
    session_id: str
    user_id: str
    window_start_turn: int = 0
    window_end_turn: int = 0
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extractor_version: str = "4.0"


class SessionObservationState(BaseModel):
    session_meta: SessionMeta
    visible_summary: str = Field("", description="轨迹摘要：2-4 句人话")
    user_interest: UserInterestState = Field(default_factory=UserInterestState)
    exploration_coordinates_md: str = Field("", description="探索坐标 Markdown（UI 主展示）")
    star_map_render: Optional[StarMapRenderPayload] = None
    retrieval_hints: List[RetrievalHint] = Field(default_factory=list)

    @field_validator("visible_summary")
    @classmethod
    def strip_summary(cls, v: str) -> str:
        return (v or "").strip()

    def sync_coordinates_md(self) -> None:
        self.exploration_coordinates_md = self.user_interest.exploration.to_markdown()

    def exploration_dict(self) -> Dict[str, Any]:
        ui = self.user_interest
        exp = ui.exploration
        return {
            "topics": list(ui.themes),
            "entities": list(dict.fromkeys(ui.characters + ui.imagery_discussed)),
            "chapters": list(ui.explored_chapters),
            "plot_refs": list(ui.confirmed_facts),
            "topic_focus": exp.topic_focus,
        }

    def to_storage_dict(self) -> Dict[str, Any]:
        self.sync_coordinates_md()
        return self.model_dump(mode="json")

    @classmethod
    def from_storage_dict(cls, data: Dict[str, Any]) -> "SessionObservationState":
        data = validate_no_identity_fields(dict(data))
        data.pop("_meta", None)
        ui_raw = data.get("user_interest") or {}

        if isinstance(ui_raw, dict):
            cls._migrate_user_interest(ui_raw, data)

        for k in ("formed_views", "open_questions", "interaction_mode", "exploration"):
            data.pop(k, None)

        state = cls.model_validate(data)
        state.sync_coordinates_md()
        return state

    @staticmethod
    def _migrate_user_interest(ui_raw: Dict[str, Any], data: Dict[str, Any]) -> None:
        if "exploration" not in ui_raw or not isinstance(ui_raw.get("exploration"), dict):
            exp = ExplorationCoordinates(
                topic_focus=ui_raw.get("focus") or "",
                analysis_angle=ui_raw.get("exploration_angle") or "",
                main_themes=(ui_raw.get("themes") or [])[:4],
                recent_change=ui_raw.get("understanding_change") or "稳定",
            )
            ui_raw["exploration"] = exp.model_dump()

        if "exploration" in data and isinstance(data["exploration"], dict):
            old_exp = data.pop("exploration")
            ui_raw.setdefault("characters", old_exp.get("entities") or [])
            ui_raw.setdefault("themes", old_exp.get("topics") or [])
            ui_raw.setdefault("explored_chapters", old_exp.get("chapters") or [])
            ui_raw.setdefault("confirmed_facts", old_exp.get("plot_refs") or [])

        if "open_questions" in data:
            oqs = data.get("open_questions") or []
            unresolved = [
                (q.get("question") if isinstance(q, dict) else str(q))
                for q in oqs[:2]
            ]
            ui_raw.setdefault("unresolved", unresolved)
            exp = ui_raw.get("exploration") or {}
            if isinstance(exp, dict) and not exp.get("main_themes"):
                exp["main_themes"] = (ui_raw.get("themes") or [])[:4]
                ui_raw["exploration"] = exp

        data["user_interest"] = ui_raw

    @classmethod
    def empty(cls, session_id: str, user_id: str) -> "SessionObservationState":
        return cls(session_meta=SessionMeta(session_id=session_id, user_id=user_id))


def validate_no_identity_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return data
    cleaned = dict(data)
    for key in list(cleaned.keys()):
        if key in FORBIDDEN_IDENTITY_KEYS:
            cleaned.pop(key, None)
    return cleaned
