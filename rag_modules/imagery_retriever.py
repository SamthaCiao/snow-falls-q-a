"""
意象系统检索器
"""
import logging
from typing import Dict
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class ImageryRetriever(BaseRetriever):
    """意象系统检索器"""
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """
        检索意象系统
        
        Args:
            query: 查询文本
            top_k: 返回top k个结果
        
        Returns:
            检索结果字典
        """
        logger.info("使用意象系统索引...")
        imagery_results = self.rag_system.search_imagery(query, top_k=top_k)
        
        if imagery_results:
            context_parts = []
            sources = []
            
            for i, result in enumerate(imagery_results, 1):
                element = result['element']
                content = result['content']
                score = result['score']
                logger.info(f"\n[意象结果 #{i}] (分数: {score:.3f})")
                logger.info(f"元素: {element}")
                # 已删除内容预览日志
                
                context_parts.append(content)
                sources.append({
                    "source": "意象系统",
                    "chapter": element,
                    "content": content[:500] + "..." if len(content) > 500 else content,
                    "hide_in_ui": True  # 标记为隐藏，不在UI中显示
                })
            
            return {
                'context': "\n\n".join(context_parts),
                'sources': sources,
                'score': imagery_results[0]['score'] if imagery_results else 0.0,
                'fallback': False
            }
        else:
            logger.warning("意象系统索引未找到结果，需要回退...")
            return {
                'context': '',
                'sources': [],
                'score': 0.0,
                'fallback': True  # 标记需要回退
            }
