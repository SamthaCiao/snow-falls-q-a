"""
章节摘要检索器（回退选项）
"""
import logging
from typing import Dict
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class ChapterSummaryRetriever(BaseRetriever):
    """章节摘要检索器（用于回退）"""
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """
        检索章节摘要
        
        Args:
            query: 查询文本
            top_k: 返回top k个结果
        
        Returns:
            检索结果字典
        """
        logger.info("使用章节摘要索引...")
        summary_results = self.rag_system.search_summary(query, top_k=top_k)
        
        if summary_results:
            context_parts = []
            sources = []
            
            for i, result in enumerate(summary_results, 1):
                chapter = result['chapter']
                summary = result['summary']
                logger.info(f"\n[摘要结果 #{i}]")
                logger.info(f"章节: {chapter}")
                logger.info(f"摘要: {summary[:200]}...")
                context_parts.append(f"[章节摘要: {chapter}]\n{summary}")
                sources.append({
                    "source": "章节摘要",
                    "chapter": chapter,
                    "content": summary[:200]
                })
            
            return {
                'context': "\n\n".join(context_parts),
                'sources': sources,
                'score': summary_results[0]['score'] if summary_results else 0.0,
                'fallback': False
            }
        else:
            logger.warning("摘要索引未找到结果，需要回退...")
            return {
                'context': '',
                'sources': [],
                'score': 0.0,
                'fallback': True
            }
