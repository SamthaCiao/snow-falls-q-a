"""
动态摘要生成器
"""
import logging
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class DynamicSummaryGenerator(BaseRetriever):
    """动态摘要生成器"""
    
    def __init__(self, rag_system, llm_client: OpenAI, model: str):
        """
        初始化动态摘要生成器
        
        Args:
            rag_system: RAG系统实例
            llm_client: LLM客户端
            model: 使用的模型名称
        """
        super().__init__(rag_system)
        self.llm_client = llm_client
        self.model = model
    
    def should_use(self, query: str, question_type: str) -> Tuple[bool, str]:
        """
        判断是否需要使用动态摘要
        
        Args:
            query: 用户查询
            question_type: 问题类型
        
        Returns:
            (是否需要动态摘要, 判断理由)
        """
        query_lower = query.lower()
        
        # 1. 跨章节综合分析类问题
        cross_chapter_keywords = ["对比", "比较", "区别", "关系发展", "变化过程", "演变", 
                                 "如何发展", "怎样变化", "前后对比", "不同阶段"]
        if any(kw in query_lower for kw in cross_chapter_keywords):
            return True, "跨章节综合分析类问题"
        
        # 2. 多角度对比类问题
        multi_angle_keywords = ["多个", "哪些", "分别", "各自", "不同", "各种", "多方面"]
        if any(kw in query_lower for kw in multi_angle_keywords) and len(query) > 20:
            return True, "多角度对比类问题"
        
        # 3. 预设摘要匹配度判断
        if question_type == "global_summary":
            specific_keywords = ["为什么", "如何", "怎样", "原因", "过程", "细节"]
            if any(kw in query_lower for kw in specific_keywords) and len(query) > 15:
                return True, "全局摘要无法覆盖的具体问题"
        
        # 4. 需要深度推理的问题
        reasoning_keywords = ["为什么", "原因", "导致", "影响", "结果", "后果", "意义"]
        if any(kw in query_lower for kw in reasoning_keywords) and len(query) > 25:
            return True, "需要深度推理的问题"
        
        return False, "简单问题，使用预设索引"
    
    def generate(self, query: str, retrieved_chapters: List[Dict]) -> Tuple[str, bool]:
        """
        基于检索到的章节内容，生成针对问题的动态摘要
        
        Args:
            query: 用户查询
            retrieved_chapters: 检索到的章节内容列表
        
        Returns:
            (生成的摘要, 摘要质量是否良好)
        """
        if not retrieved_chapters:
            return "", False
        
        # 按章节分组内容
        chapters_content = {}
        for item in retrieved_chapters:
            chapter = item.get('metadata', {}).get('chapter', '未知章节')
            content = item.get('content', '')
            if chapter not in chapters_content:
                chapters_content[chapter] = []
            chapters_content[chapter].append(content)
        
        # 构建章节内容文本（限制总长度）
        chapters_text = []
        total_length = 0
        max_length = 4000
        
        for chapter, contents in chapters_content.items():
            chapter_text = f"\n## {chapter}\n" + "\n\n".join(contents[:3])
            if total_length + len(chapter_text) > max_length:
                break
            chapters_text.append(chapter_text)
            total_length += len(chapter_text)
        
        combined_text = "\n".join(chapters_text)
        
        # 构建摘要生成提示词
        summary_prompt = f"""你是一个专业的文本摘要专家。请基于以下小说章节内容，针对用户的问题生成一个聚焦的上下文摘要。

用户问题：{query}

相关章节内容：
{combined_text}

要求：
1. 摘要必须聚焦于回答用户问题的关键信息
2. 保持信息的准确性和完整性
3. 如果涉及多个章节，请按逻辑顺序组织信息
4. 摘要长度控制在500-800字
5. 只返回摘要内容，不要添加额外说明

请生成摘要："""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文本摘要专家，擅长从长文本中提取关键信息并生成聚焦的摘要。"},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            summary = response.choices[0].message.content.strip()
            
            # 验证摘要质量
            if len(summary) < 100:
                logger.warning("动态摘要过短，质量可能不足")
                return summary, False
            
            # 检查是否包含查询中的关键词
            query_keywords = [w for w in query.split() if len(w) > 1]
            matched_keywords = sum(1 for kw in query_keywords if kw.lower() in summary.lower())
            if matched_keywords < len(query_keywords) * 0.3:
                logger.warning("动态摘要可能未充分覆盖查询关键词")
                return summary, False
            
            logger.info(f"动态摘要生成成功，长度: {len(summary)} 字符")
            return summary, True
            
        except Exception as e:
            logger.error(f"动态摘要生成失败: {e}")
            return "", False
    
    def retrieve(self, query: str, top_k: int = 20) -> Dict:
        """
        生成动态摘要（如果适用）
        
        Args:
            query: 查询文本
            top_k: 检索的章节数量
        
        Returns:
            检索结果字典
        """
        # 先检索相关章节
        search_results = self.rag_system.search(
            query=query,
            top_k=top_k,
            use_background=True,
            use_hybrid=True
        )
        
        # 提取检索结果
        retrieved_chapters = []
        if search_results['target_source'] == "影化成殇":
            retrieved_chapters = search_results['sequel_results']
        else:
            retrieved_chapters = search_results['main_results']
        
        if retrieved_chapters:
            logger.info(f"检索到 {len(retrieved_chapters)} 个相关章节片段")
            
            # 生成动态摘要
            dynamic_summary, summary_quality_good = self.generate(query, retrieved_chapters)
            
            if summary_quality_good and dynamic_summary:
                logger.info("动态摘要生成成功，使用动态摘要作为上下文")
                return {
                    'context': f"[动态摘要（针对问题生成）]\n{dynamic_summary}",
                    'sources': [{
                        "source": "动态摘要",
                        "chapter": "",
                        "content": dynamic_summary[:500] + "..." if len(dynamic_summary) > 500 else dynamic_summary
                    }],
                    'target_source': search_results.get('target_source', '雪落成诗'),
                    'score': 1.0,
                    'fallback': False,
                    'used_dynamic_summary': True
                }
        
        # 如果动态摘要质量不足，返回空结果，让主程序回退
        return {
            'context': '',
            'sources': [],
            'score': 0.0,
            'fallback': True,
            'used_dynamic_summary': False
        }
