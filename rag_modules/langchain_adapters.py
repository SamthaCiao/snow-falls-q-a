"""
LangChain 适配层
将现有检索器封装为 LangChain BaseRetriever，支持链式编排与可观测性。
保留现有多路、多跳、行为/情节逻辑，仅标准化检索接口与编排。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Any, Callable, Optional

logger = logging.getLogger(__name__)

# 可选依赖：无 LangChain 时仍可运行（走原有逻辑）
try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    BaseRetriever = object  # type: ignore
    Document = None  # type: ignore
    CallbackManagerForRetrieverRun = None  # type: ignore


def _dict_to_documents(result: Dict[str, Any]) -> List["Document"]:
    """将现有检索器返回的 dict 转为 LangChain Document 列表，便于链式编排与观测。"""
    if not HAS_LANGCHAIN or Document is None:
        return []
    context = result.get("context", "") or ""
    sources = result.get("sources", [])
    target_source = result.get("target_source", "")
    fallback = result.get("fallback", True)
    score = result.get("score", 0.0)
    metadata: Dict[str, Any] = {
        "sources": sources,
        "target_source": target_source,
        "fallback": fallback,
        "score": score,
    }
    if not context.strip():
        return []
    return [Document(page_content=context.strip(), metadata=metadata)]


def documents_to_result(docs: List[Any]) -> Dict[str, Any]:
    """将 LangChain Document 列表（或具 page_content/metadata 的对象列表）转回现有系统使用的 result 字典。"""
    if not docs:
        return {
            "context": "",
            "sources": [],
            "target_source": "雪落成诗",
            "fallback": True,
            "score": 0.0,
        }
    doc = docs[0]
    page_content = getattr(doc, "page_content", "") or ""
    meta = getattr(doc, "metadata", None) or {}
    return {
        "context": page_content,
        "sources": meta.get("sources", []),
        "target_source": meta.get("target_source", "雪落成诗"),
        "fallback": meta.get("fallback", True),
        "score": meta.get("score", 0.0),
    }


if HAS_LANGCHAIN:

    class LangChainRetrieverAdapter(BaseRetriever):
        """
        将现有检索器（retrieve(query) -> Dict）封装为 LangChain BaseRetriever。
        返回 List[Document]，metadata 中保留 sources、target_source、fallback 供上层使用。
        """

        retrieve_fn: Callable[..., Dict[str, Any]]
        """调用约定：retrieve_fn(query, **kwargs) -> Dict 与现有检索器一致"""
        fn_kwargs: Dict[str, Any] = {}
        """传给 retrieve_fn 的额外关键字参数（如 question_type）"""

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        ) -> List[Document]:
            kwargs = dict(self.fn_kwargs)
            try:
                result = self.retrieve_fn(query, **kwargs)
            except TypeError:
                result = self.retrieve_fn(query)
            return _dict_to_documents(result)


def build_langchain_retriever_map(
    retrievers: Dict[str, Any],
) -> Dict[str, Any]:
    """
    为现有 self.retrievers 字典中的每个检索器构建 LangChain BaseRetriever 包装。
    用于链式编排与统一调用；行为/情节类检索器需要 question_type 等参数，通过 fn_kwargs 传入。
    """
    if not HAS_LANGCHAIN:
        return {}

    lc_map: Dict[str, Any] = {}
    for key, retriever in retrievers.items():
        if key in ("behavior_explanation", "scenario_simulation"):
            lc_map[key] = LangChainRetrieverAdapter(
                retrieve_fn=retriever.retrieve,
                fn_kwargs={"question_type": key},
            )
        else:
            lc_map[key] = LangChainRetrieverAdapter(
                retrieve_fn=retriever.retrieve,
                fn_kwargs={},
            )
    return lc_map
