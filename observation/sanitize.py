"""观测 state 后处理。"""
from __future__ import annotations

import re
from typing import List, Optional, Set

from observation.character_index import get_known_entity_names
from observation.schema import ExplorationCoordinates, RetrievalHint, SessionObservationState

_FRAGMENT_DENY = frozenset({
    "什么", "怎么", "为什么", "如何", "是否", "一个", "一种", "这部", "这本",
    "整体", "第一次", "关键", "节点", "如果", "可能", "会怎", "么故", "键节",
    "点选", "一次产", "如果雪", "可能会", "雪遗诗和", "为什么夏", "和大学",
    "诗没有", "了一个",
})

_BAD_SUMMARY = re.compile(
    r"(展开了\s*\d+\s*轮|仍有\s*\d+\s*个|待继续深入|么故事|键节点|新探索)"
)

# prompt / 用户示例里出现、不应出现在真实输出中的措辞
_PROMPT_EXAMPLE_LEAK = (
    "第7章沉默",
    "第七集沉默",
    "逃避责任",
    "沉默作为逃避",
    "暗恋与权力",
    "从外部环境转向内在性格",
    "当时在场其他人",
    "试试从「当时在场其他人",
    "倾向「逃避责任」",
    "心理动机分析，倾向",
)


def detect_prompt_example_leak(state: SessionObservationState) -> bool:
    """检测 LLM 是否照搬 prompt/示例，而非按对话生成。"""
    exp = state.user_interest.exploration
    blob = " ".join([
        state.visible_summary or "",
        exp.topic_focus or "",
        exp.analysis_angle or "",
        exp.recent_change or "",
        exp.pivot_suggestion or "",
        " ".join(exp.main_themes or []),
    ])
    return any(p in blob for p in _PROMPT_EXAMPLE_LEAK)


def detect_stale_trajectory(state: SessionObservationState) -> bool:
    """内容级旧版/乱码检测（不依赖版本号）。"""
    if detect_prompt_example_leak(state):
        return True
    if _BAD_SUMMARY.search(state.visible_summary or ""):
        return True
    return False


def _is_garbage_token(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) < 2:
        return True
    if t in _FRAGMENT_DENY:
        return True
    if len(t) <= 4 and re.fullmatch(r"[\u4e00-\u9fff]{2,4}", t):
        return True
    return False


def _match_known_name(text: str, known: Set[str]) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    if t in known:
        return t
    for k in sorted(known, key=len, reverse=True):
        if len(k) >= 2 and k in t:
            return k
    return None


def _filter_characters(chars: List[str], known: Set[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in chars:
        name = _match_known_name(raw, known)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
        if len(out) >= 6:
            break
    return out


def _dedupe(items: List[str], max_n: int) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        x = (x or "").strip()
        if not x or x in seen or _is_garbage_token(x):
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= max_n:
            break
    return out


def _sanitize_summary(text: str) -> str:
    t = (text or "").strip()
    if not t or _BAD_SUMMARY.search(t):
        return ""
    if "、" in t and len(t) < 80 and sum(1 for p in t.split("、") if len(p) <= 4) >= 3:
        return ""
    return t


def _dedupe_summary_vs_coordinates(
    summary: str,
    exp: ExplorationCoordinates,
    user_messages: List[str],
) -> str:
    """若摘要与坐标字段高度重合，去掉摘要中重复的主题词堆砌句。"""
    themes = set(exp.main_themes or [])
    if exp.topic_focus:
        themes.add(exp.topic_focus)
    overlap = sum(1 for th in themes if th and th in summary)
    if overlap >= 2 and len(summary) < 120:
        for th in themes:
            if th and len(th) >= 2:
                summary = summary.replace(f"「{th}」", "").replace(f"{th}、", "").replace(f"、{th}", "")
        summary = re.sub(r"\s{2,}", " ", summary).strip(" ，。")
    return summary or _journey_fallback(user_messages, [], exp)


def _journey_fallback(
    user_messages: List[str],
    characters: List[str],
    exp: ExplorationCoordinates,
) -> str:
    """叙述式兜底，不复述坐标关键词。"""
    if len(user_messages) >= 2:
        first = (user_messages[0] or "")[:20].rstrip("？?")
        last = (user_messages[-1] or "")[:20].rstrip("？?")
        if first != last:
            return f"你起初从「{first}…」切入，随后转向「{last}…」，阅读路径在逐步展开。"
    if characters:
        return f"你的提问目前主要围绕「{characters[0]}」相关情节展开，探索仍在继续。"
    return "你已开始与作品对话，阅读轨迹随提问逐步成形。"


def merge_user_interest_from_previous(
    state: SessionObservationState,
    previous: SessionObservationState,
) -> SessionObservationState:
    """全量重抽取时保留已探索人物/意象，避免 LLM 漏填导致星图归零。"""
    pu = previous.user_interest
    nu = state.user_interest
    nu.characters = list(dict.fromkeys(list(nu.characters) + list(pu.characters)))[:8]
    nu.imagery_discussed = list(dict.fromkeys(list(nu.imagery_discussed) + list(pu.imagery_discussed)))[:8]
    nu.themes = list(dict.fromkeys(list(nu.themes) + list(pu.themes)))[:6]
    nu.explored_chapters = list(dict.fromkeys(list(nu.explored_chapters) + list(pu.explored_chapters)))[:8]
    if not nu.exploration.topic_focus and pu.exploration.topic_focus:
        nu.exploration.topic_focus = pu.exploration.topic_focus
    if not nu.exploration.main_themes and pu.exploration.main_themes:
        nu.exploration.main_themes = list(pu.exploration.main_themes)[:4]
    state.exploration_coordinates_md = nu.exploration.to_markdown()
    return state


def enrich_characters_from_turns(
    state: SessionObservationState,
    user_messages: List[str],
) -> SessionObservationState:
    """从用户原话扫描已知人物名，作为 characters 兜底。"""
    known = set(get_known_entity_names())
    if not known or not user_messages:
        return state
    blob = "\n".join(user_messages)
    skip = {"雪落成诗", "影化成殇"}
    found: List[str] = []
    for name in sorted(known, key=len, reverse=True):
        if name in skip or len(name) < 2 or name not in blob or name in found:
            continue
        found.append(name)
        if len(found) >= 8:
            break
    if found:
        ui = state.user_interest
        ui.characters = list(dict.fromkeys(list(ui.characters) + found))[:8]
    return state


def sanitize_observation_state(
    state: SessionObservationState,
    *,
    user_messages: Optional[List[str]] = None,
) -> SessionObservationState:
    known = set(get_known_entity_names())
    ui = state.user_interest
    exp = ui.exploration

    ui.characters = _filter_characters(ui.characters, known)
    ui.themes = _dedupe(ui.themes, 4)
    ui.explored_chapters = _dedupe(
        [c for c in ui.explored_chapters if c and len(c.strip()) >= 2],
        8,
    )
    ui.confirmed_facts = _dedupe(
        [f for f in ui.confirmed_facts if f and len(f.strip()) >= 4],
        6,
    )
    ui.imagery_discussed = _dedupe(ui.imagery_discussed, 8)

    exp.main_themes = _dedupe(exp.main_themes or ui.themes, 4)
    if not exp.main_themes and ui.themes:
        exp.main_themes = _dedupe(ui.themes, 4)
    if not exp.main_themes and exp.topic_focus:
        exp.main_themes = [exp.topic_focus[:12]]
    if exp.topic_focus:
        tf = exp.topic_focus.strip()
        if len(tf) < 2 or tf in _FRAGMENT_DENY:
            exp.topic_focus = ""
        else:
            exp.topic_focus = tf[:10]
    if exp.analysis_angle and len(exp.analysis_angle) > 80:
        exp.analysis_angle = exp.analysis_angle[:77] + "…"
    if exp.recent_change and len(exp.recent_change) > 80:
        exp.recent_change = exp.recent_change[:77] + "…"
    if exp.pivot_suggestion and len(exp.pivot_suggestion) > 100:
        exp.pivot_suggestion = exp.pivot_suggestion[:97] + "…"

    if not ui.characters and exp.topic_focus:
        for k in sorted(known, key=len, reverse=True):
            if len(k) >= 2 and k in exp.topic_focus:
                ui.characters.append(k)
                break

    clean_summary = _sanitize_summary(state.visible_summary)
    if not clean_summary:
        state.visible_summary = _journey_fallback(user_messages or [], ui.characters, exp)
    else:
        state.visible_summary = _dedupe_summary_vs_coordinates(clean_summary, exp, user_messages or [])

    hints: List[RetrievalHint] = []
    for h in state.retrieval_hints:
        q = (h.query or "").strip()
        if q and not _is_garbage_token(q):
            hints.append(h)
        if len(hints) >= 5:
            break
    state.retrieval_hints = hints

    state.exploration_coordinates_md = exp.to_markdown()
    return state
