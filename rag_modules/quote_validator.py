"""
回答幻觉校验：引文逐字核对 + 无引号事实句 grounding 检查。
未通过时供上层打回重写；仍失败则剥离无效引号。
"""
import re
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# 从上下文中截取「可逐字引用」的语料块
_VERBATIM_MARKERS = [
    "【2. RAG检索到的原文片段】",
    "[文档检索 - 原文片段]",
    "[与时间线/人物/事件链对应的原文]",
    "【作品原文补充】",
    "[来源:",
]

_QUOTE_PATTERNS = [
    re.compile(r"「([^」]{2,})」"),
    re.compile(r'"([^"]{2,})"'),
    re.compile(r"“([^”]{2,})”"),
    re.compile(r"'([^']{2,})'"),
]

_ANALYTICAL_MARKERS = (
    "可能", "似乎", "也许", "推测", "或许", "大概", "主观", "可以理解为",
    "倾向于", "不排除", "一种解读", "从文本看", "值得注意的是",
)

_FACTUAL_HINT = re.compile(
    r"(?:在|于|发生了|进行了|导致|随后|之后|之前|第.+章|高[一二三]|大学|高考|喜欢|拒绝|离开|出现)"
)

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；\n])")


def extract_verbatim_corpus(context: str) -> str:
    """从检索上下文中提取允许逐字引用的原文语料。"""
    if not context:
        return ""
    parts: List[str] = []
    for marker in _VERBATIM_MARKERS:
        idx = context.find(marker)
        if idx < 0:
            continue
        start = idx + len(marker)
        rest = context[start:]
        next_block = re.search(
            r"\n(?:【\d+\.|【\d|【[0-9]|\[人物|\[时间线|\[关键事件|\[文档检索|\[与)",
            rest,
        )
        chunk = rest[: next_block.start()] if next_block else rest
        parts.append(chunk.strip())
    if parts:
        return "\n".join(parts)
    if "心理模型" not in context and "【1. 时间线索引】" not in context:
        return context
    return ""


def extract_grounding_corpus(context: str) -> str:
    """事实 grounding 语料：原文 + 时间线 + 结构化检索结果（整段 context）。"""
    return (context or "").strip()


def _normalize_for_match(text: str) -> str:
    t = re.sub(r"\s+", "", text or "")
    t = re.sub(r"[，。！？、；：""''「」『』…—·-]", "", t)
    return t


def extract_quoted_spans(answer: str, min_len: int = 4) -> List[str]:
    """提取回答中的引文片段（去重）。"""
    if not answer:
        return []
    found: List[str] = []
    seen = set()
    for pat in _QUOTE_PATTERNS:
        for m in pat.finditer(answer):
            s = (m.group(1) or "").strip()
            if len(s) < min_len:
                continue
            key = _normalize_for_match(s)
            if key and key not in seen:
                seen.add(key)
                found.append(s)
    return found


def quote_in_corpus(quote: str, corpus: str) -> bool:
    """引文是否在语料中（归一化子串匹配，允许标点差异）。"""
    nq = _normalize_for_match(quote)
    nc = _normalize_for_match(corpus)
    if not nq or not nc:
        return False
    if nq in nc:
        return True
    if len(nq) >= 10:
        window = max(6, int(len(nq) * 0.85))
        for i in range(0, len(nq) - window + 1):
            if nq[i: i + window] in nc:
                return True
    return False


def validate_answer_quotes(
    answer: str,
    context: str,
    *,
    min_quote_len: int = 4,
) -> Tuple[bool, List[str], Optional[str]]:
    """
    校验回答中的引文。

    Returns:
        (passed, invalid_quotes, verbatim_corpus)
    """
    corpus = extract_verbatim_corpus(context)
    quotes = extract_quoted_spans(answer, min_len=min_quote_len)
    if not quotes:
        return True, [], corpus or None
    if not corpus.strip():
        return False, quotes, None
    invalid = [q for q in quotes if not quote_in_corpus(q, corpus)]
    return len(invalid) == 0, invalid, corpus


def _extract_terms(text: str) -> List[str]:
    try:
        import jieba
        return [w.strip() for w in jieba.lcut(text) if len(w.strip()) >= 2]
    except ImportError:
        return re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", text)


def _token_overlap(sentence: str, corpus: str) -> float:
    if not sentence or not corpus:
        return 0.0
    ts = _extract_terms(sentence)
    if not ts:
        return 1.0
    nc = _normalize_for_match(corpus)
    hit = sum(1 for t in ts if _normalize_for_match(t) in nc)
    return hit / len(ts)


def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT.split(text or "")
    return [p.strip() for p in parts if p and p.strip()]


def validate_factual_grounding(
    answer: str,
    context: str,
    *,
    min_overlap: float = 0.18,
    min_sentence_len: int = 10,
) -> Tuple[bool, List[str]]:
    """
    校验无引号的事实性陈述是否与 grounding 语料有关。
    分析性/推测性句子跳过；过短句子跳过。
    """
    corpus = extract_grounding_corpus(context)
    if not corpus.strip() or not (answer or "").strip():
        return True, []

    invalid: List[str] = []
    for sent in _split_sentences(answer):
        if len(sent) < min_sentence_len:
            continue
        if any(m in sent for m in _ANALYTICAL_MARKERS):
            continue
        if "「" in sent or "」" in sent or '"' in sent or "“" in sent:
            continue
        terms = _extract_terms(sent)
        is_factual_candidate = bool(_FACTUAL_HINT.search(sent)) or len(terms) >= 5
        if not is_factual_candidate:
            continue
        overlap = _token_overlap(sent, corpus)
        if overlap < min_overlap:
            invalid.append(sent)

    return len(invalid) == 0, invalid[:5]


def validate_answer(
    answer: str,
    context: str,
    *,
    min_quote_len: int = 4,
    check_facts: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """综合校验：引文 + 可选事实 grounding。"""
    quote_ok, invalid_quotes, _ = validate_answer_quotes(
        answer, context, min_quote_len=min_quote_len
    )
    invalid_facts: List[str] = []
    if check_facts and context and context.strip():
        fact_ok, invalid_facts = validate_factual_grounding(answer, context)
    else:
        fact_ok = True
    return quote_ok and fact_ok, {
        "invalid_quotes": invalid_quotes,
        "invalid_facts": invalid_facts,
    }


def sanitize_invalid_quotes(answer: str, invalid_quotes: List[str]) -> str:
    """移除无法验证的引号包裹，保留文字（转述）。"""
    if not answer or not invalid_quotes:
        return answer
    result = answer
    for q in invalid_quotes:
        for left, right in (("「", "」"), ('"', '"'), ("“", "”"), ("'", "'")):
            wrapped = f"{left}{q}{right}"
            if wrapped in result:
                result = result.replace(wrapped, q)
    return result


def append_grounding_disclaimer(answer: str, invalid_facts: List[str]) -> str:
    """对仍无法 grounding 的事实句追加简短提示（不阻断回答）。"""
    if not invalid_facts:
        return answer
    note = (
        "\n\n---\n"
        "*以上分析中部分具体情节细节未在检索片段中直接出现，"
        "请以作品原文为准；如需核实可继续追问具体章节。*"
    )
    if note.strip() in answer:
        return answer
    return answer.rstrip() + note
