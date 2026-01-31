"""
章节RAG检索器（混合检索）
"""
import logging
from typing import Dict
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class ChapterRAGRetriever(BaseRetriever):
    """章节RAG检索器（使用混合检索）"""
    
    def retrieve(self, query: str, top_k: int = 10) -> Dict:
        """
        使用混合检索（语义向量 + BM25）检索章节内容
        
        Args:
            query: 查询文本
            top_k: 返回top k个结果
        
        Returns:
            检索结果字典
        """
        logger.info("使用混合检索（语义向量 + BM25）...")
        
        # 执行混合检索
        search_results = self.rag_system.search(
            query=query,
            top_k=top_k,
            use_background=True,
            use_hybrid=True
        )
        
        # 记录检索结果
        logger.info(f"\n检索到 {len(search_results.get('main_results', []))} 个主文档结果")
        logger.info(f"检索到 {len(search_results.get('sequel_results', []))} 个续集文档结果")
        
        target_source = search_results['target_source']
        context_parts = []
        sources = []
        
        # 跨文档检索：关联内容兼顾主文档和续集；续集问题优先采用续集文本
        if target_source == "影化成殇":
            # 续集问题：优先续集结果（更多条），再补充主文档背景（较少条）
            for i, result in enumerate(search_results['sequel_results'][:6], 1):
                content = result['content']
                chapter = result['metadata'].get('chapter', '未知章节')
                score = result.get('combined_score', 0.0)
                logger.info(f"\n[续集结果 #{i}] (综合分数: {score:.3f})")
                logger.info(f"章节: {chapter}")
                context_parts.append(f"[来源: 影化成殇 - {chapter}]\n{content}")
                sources.append({
                    "source": "影化成殇",
                    "chapter": chapter,
                    "content": content[:200]
                })
            for i, result in enumerate(search_results.get('background_results', [])[:3], 1):
                content = result['content']
                chapter = result.get('metadata', {}).get('chapter', '未知章节')
                logger.info(f"\n[主文档关联 #{i}] 章节: {chapter}")
                context_parts.append(f"[来源: 雪落成诗 - {chapter}]\n{content}")
                sources.append({
                    "source": "雪落成诗",
                    "chapter": chapter,
                    "content": content[:200]
                })
        else:
            # 主文档问题：优先主文档结果，再补充续集关联
            for i, result in enumerate(search_results['main_results'][:6], 1):
                content = result['content']
                chapter = result['metadata'].get('chapter', '未知章节')
                score = result.get('combined_score', 0.0)
                logger.info(f"\n[主文档结果 #{i}] (综合分数: {score:.3f})")
                logger.info(f"章节: {chapter}")
                context_parts.append(f"[来源: 雪落成诗 - {chapter}]\n{content}")
                sources.append({
                    "source": "雪落成诗",
                    "chapter": chapter,
                    "content": content[:200]
                })
            for i, result in enumerate(search_results.get('background_results', [])[:3], 1):
                content = result['content']
                chapter = result.get('metadata', {}).get('chapter', '未知章节')
                logger.info(f"\n[续集关联 #{i}] 章节: {chapter}")
                context_parts.append(f"[来源: 影化成殇 - {chapter}]\n{content}")
                sources.append({
                    "source": "影化成殇",
                    "chapter": chapter,
                    "content": content[:200]
                })
        
        return {
            'context': "\n\n---\n\n".join(context_parts),
            'sources': sources,
            'target_source': target_source or "雪落成诗",
            'score': search_results['sequel_results'][0].get('combined_score', 0.0) if target_source == "影化成殇" and search_results['sequel_results'] else (search_results['main_results'][0].get('combined_score', 0.0) if search_results['main_results'] else 0.0),
            'fallback': False
        }
