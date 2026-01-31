"""
时间线检索器
"""
import logging
from typing import Dict
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class TimelineRetriever(BaseRetriever):
    """时间线检索器"""
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """
        检索时间线
        
        Args:
            query: 查询文本
            top_k: 返回top k个结果
        
        Returns:
            检索结果字典
        """
        logger.info("使用时间线索引...")
        timeline_results = self.rag_system.search_timeline(query, top_k=top_k)
        
        if timeline_results:
            context_parts = []
            sources = []
            preferred_chapters = []
            for i, result in enumerate(timeline_results, 1):
                timepoint = result['timepoint']
                events = result['events']
                chapters = result.get('chapters', [])
                if chapters:
                    preferred_chapters.extend(chapters)
                logger.info(f"\n[时间线结果 #{i}] (分数: {result['score']:.3f})")
                logger.info(f"时间点: {timepoint}")
                logger.info(f"事件数: {len(events)}")
                
                context_parts.append(f"[时间线: {timepoint}]\n" + "\n".join([f"- {e}" for e in events]))
                sources.append({
                    "source": "时间线",
                    "chapter": timepoint,
                    "content": f"{timepoint}: " + "; ".join(events[:3]),
                    "hide_in_ui": True
                })
            preferred_chapters = list(dict.fromkeys(preferred_chapters))
            return {
                'context': "\n\n".join(context_parts),
                'sources': sources,
                'score': timeline_results[0]['score'] if timeline_results else 0.0,
                'fallback': False,
                'preferred_chapters': preferred_chapters,
            }
        else:
            logger.warning("时间线索引未找到结果，需要回退...")
            return {
                'context': '',
                'sources': [],
                'score': 0.0,
                'fallback': True,
                'preferred_chapters': [],
            }
