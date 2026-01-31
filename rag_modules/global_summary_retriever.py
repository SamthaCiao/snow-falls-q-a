"""
全局摘要检索器
"""
import logging
from typing import Dict
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class GlobalSummaryRetriever(BaseRetriever):
    """全局摘要检索器"""
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """
        检索全局摘要
        
        Args:
            query: 查询文本
            top_k: 返回top k个结果（全局摘要通常只有一个）
        
        Returns:
            检索结果字典
        """
        logger.info("使用全局摘要索引...")
        global_result = self.rag_system.search_global_summary(query)
        
        if global_result['content']:
            logger.info(f"\n[全局摘要]")
            logger.info(f"内容长度: {len(global_result['content'])} 字符")
            logger.info(f"匹配分数: {global_result['score']:.3f}")
            # 已删除摘要预览日志
            
            return {
                'context': f"[全局摘要]\n{global_result['content']}",
                'sources': [{
                    "source": "全局摘要",
                    "chapter": "",
                    "content": global_result['content'][:500] + "..." if len(global_result['content']) > 500 else global_result['content']
                }],
                'score': global_result['score'],
                'fallback': False
            }
        else:
            logger.warning("全局摘要为空，需要回退...")
            return {
                'context': '',
                'sources': [],
                'score': 0.0,
                'fallback': True  # 标记需要回退
            }
