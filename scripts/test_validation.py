"""校验与结构化查询单元测试（本地 dev，无需 API）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_quote_and_fact_validation():
    from rag_modules.quote_validator import (
        validate_answer,
        validate_answer_quotes,
        sanitize_invalid_quotes,
    )

    ctx = (
        "【1. 时间线索引】（最高权威事实）\n[时间点: 高三]\n- 雪遗诗与夏空山看电影\n\n"
        "【2. RAG检索到的原文片段】（用于事实核查）\n"
        "雪遗诗站在窗边，看着外面的雨。夏空山没有说话。"
    )
    ok, _, _ = validate_answer_quotes("她说「雪遗诗站在窗边」", ctx)
    assert ok, "valid quote should pass"
    bad, inv, _ = validate_answer_quotes("她说「完全不存在的幻觉句子内容」", ctx)
    assert not bad and inv, "invalid quote should fail"

    factual_ok, issues = validate_answer(
        "高三时雪遗诗与夏空山一起看了电影，这是关键转折。",
        ctx,
    )
    assert factual_ok, issues

    factual_bad, issues2 = validate_answer(
        "大一寒假他们一起去哈尔滨滑雪并见了家长。",
        ctx,
    )
    assert not factual_bad and issues2.get("invalid_facts"), issues2

    stripped = sanitize_invalid_quotes("引「完全不存在的幻觉句子内容」", inv)
    assert "「" not in stripped or "完全不存在的幻觉句子内容" in stripped
    print("  OK  quote + fact validation")


def test_structured_query_no_full_json():
    from rag_modules.structured_query_engine import StructuredQueryEngine

    engine = StructuredQueryEngine()
    rules = {
        "rules": [
            {"rule_id": "R001", "类型": "人物一致性", "描述": "行为符合性格", "示例": "雪遗诗被拒绝时更可能讨好", "应用方式": "检查"},
            {"rule_id": "R002", "类型": "环境约束", "描述": "考虑学校限制", "示例": "高中不能同居", "应用方式": "约束"},
        ]
    }
    out = engine.query_index("scenario", "雪遗诗 讨好", rules, max_results=2)
    assert out and "json" not in out.lower()
    assert "R001" not in out or "人物一致性" in out
    assert len(out) < 800, "should be excerpt not full dump"

    chars = {
        "雪遗诗": {
            "role": "主角",
            "key_traits": ["敏感", "执着"],
            "心理模型": {"核心需求": "被看见", "行为模式": ["讨好", "回避"]},
            "key_events": [{"event": "与夏空山看电影", "source": "1.2"}],
        }
    }
    char_out = engine.query_index("character", "雪遗诗 讨好", chars, max_results=1)
    assert char_out and "敏感" in char_out
    assert char_out.count("{") < 3, "should not dump raw JSON blocks"
    print("  OK  structured query character/scenario")


def main():
    print("=== Unit Tests ===")
    test_quote_and_fact_validation()
    test_structured_query_no_full_json()
    print("=== All passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
