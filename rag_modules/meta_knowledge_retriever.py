"""
元文本知识库检索器
"""
import os
import json
import logging
import re
from typing import Dict, List, Optional
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class MetaKnowledgeRetriever(BaseRetriever):
    """元文本知识库检索器"""
    
    def __init__(self, rag_system):
        """初始化元文本知识库检索器"""
        super().__init__(rag_system)
        self._data_dir = getattr(rag_system, "data_dir", "数据源")
        # 先尝试修复文件（如果包含多个JSON对象）
        self._fix_meta_knowledge_file_if_needed()
        self.meta_knowledge = self._load_meta_knowledge()
        if self.meta_knowledge:
            logger.info("元文本知识库加载完成")
        else:
            logger.warning("元文本知识库加载失败或为空")
    
    def _fix_meta_knowledge_file_if_needed(self):
        """如果需要，修复元文本知识库文件（将多个JSON对象合并为一个）"""
        meta_file = os.path.join(self._data_dir, "元文本知识库.json")
        if not os.path.exists(meta_file):
            return
        
        try:
            # 读取文件内容
            with open(meta_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否有多个根级别的JSON对象（简单检查：多个独立的 { ... }）
            first_brace = content.find('{')
            if first_brace == -1:
                return
            
            # 检查第二个独立的 { 是否出现在第一个对象结束后
            depth = 0
            found_second_root = False
            for i, c in enumerate(content[first_brace:], start=first_brace):
                if c == '{':
                    if depth == 0 and i > first_brace:
                        found_second_root = True
                        break
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0 and i < len(content) - 1:
                        # 检查下一个非空白字符是否是 {
                        next_char_idx = i + 1
                        while next_char_idx < len(content) and content[next_char_idx].isspace():
                            next_char_idx += 1
                        if next_char_idx < len(content) and content[next_char_idx] == '{':
                            found_second_root = True
                            break
            
            if not found_second_root:
                # 文件格式正确，无需修复
                return
            
            # 需要修复：解析多个JSON对象并合并
            logger.info("检测到元文本知识库文件包含多个JSON对象，正在修复...")
            objs = []
            start = 0
            depth = 0
            
            for i, c in enumerate(content):
                if c == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(content[start:i+1])
                            objs.append(obj)
                        except json.JSONDecodeError:
                            pass
            
            if len(objs) <= 1:
                return
            
            # 合并所有对象
            merged = {}
            for obj in objs:
                for key, value in obj.items():
                    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key] = self._deep_merge_dict(merged[key], value)
                    else:
                        merged[key] = value
            
            # 备份原文件
            backup_path = meta_file + ".backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"原文件已备份到: {backup_path}")
            
            # 写入修复后的文件
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已修复元文本知识库文件，合并了 {len(objs)} 个JSON对象")
            
        except Exception as e:
            logger.warning(f"修复元文本知识库文件时出错（将使用代码层面的修复）: {e}")
    
    def _load_meta_knowledge(self) -> Optional[Dict]:
        """加载元文本知识库"""
        meta_file = os.path.join(self._data_dir, "元文本知识库.json")
        if not os.path.exists(meta_file):
            logger.warning(f"元文本知识库文件不存在: {meta_file}")
            return None
        
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析多个JSON对象并合并
            objs = []
            start = 0
            depth = 0
            
            for i, c in enumerate(content):
                if c == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(content[start:i+1])
                            objs.append(obj)
                        except json.JSONDecodeError as e:
                            logger.warning(f"解析JSON对象失败（位置 {start}-{i+1}）: {e}")
                            pass
            
            # 合并所有对象
            merged = {}
            for obj in objs:
                for key, value in obj.items():
                    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key] = self._deep_merge_dict(merged[key], value)
                    else:
                        merged[key] = value
            
            logger.info(f"成功加载元文本知识库，包含 {len(objs)} 个JSON对象")
            return merged
            
        except Exception as e:
            logger.error(f"加载元文本知识库失败: {e}")
            return None
    
    def _deep_merge_dict(self, dict1: Dict, dict2: Dict) -> Dict:
        """深度合并两个字典"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dict(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + [item for item in value if item not in result[key]]
            else:
                result[key] = value
        return result
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """
        从元文本知识库中检索相关信息
        
        Args:
            query: 用户问题
            top_k: 返回top k个结果
        
        Returns:
            检索结果字典
        """
        if not self.meta_knowledge:
            return {
                'content': '',
                'sources': [],
                'score': 0.0,
                'fallback': True
            }
        
        logger.info("使用元文本知识库检索...")
        results = []
        query_keywords = self._extract_meta_keywords(query)
        preferred_sections = self._meta_sections_for_query(query)
        
        # 递归搜索元文本知识库
        def search_recursive(data, path="", depth=0):
            """递归搜索JSON结构"""
            if depth > 10:  # 限制递归深度
                return
            
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    key_lower = key.lower()
                    score = 0.0
                    
                    # 关键词匹配
                    for kw in query_keywords:
                        if kw in key_lower:
                            score += 2.0
                    
                    # 检查值是否匹配
                    if isinstance(value, str):
                        value_lower = value.lower()
                        for kw in query_keywords:
                            if kw in value_lower:
                                score += 1.0
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and any(kw in item.lower() for kw in query_keywords):
                                score += 0.5
                    
                    if score > 0:
                        result_text = self._format_meta_result(key, value, current_path)
                        results.append({
                            'path': current_path,
                            'content': result_text,
                            'score': score
                        })
                    
                    # 递归搜索
                    search_recursive(value, current_path, depth + 1)
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    search_recursive(item, f"{path}[{i}]", depth + 1)
        
        # 开始搜索
        search_recursive(self.meta_knowledge)
        
        # 按问题意图优先相关段落：优先展示 preferred_sections 下的结果，再按分数排序
        def sort_key(r):
            path = (r.get('path') or '')
            is_preferred = any(path.startswith(s) for s in preferred_sections) if preferred_sections else False
            return (0 if is_preferred else 1, -r['score'])
        results.sort(key=sort_key)
        
        # 提取前top_k个最相关的结果
        top_results = results[:top_k]
        
        if top_results:
            # 合并内容
            combined_content = "\n\n".join([r['content'] for r in top_results])
            logger.info(f"\n[元文本结果] (分数: {top_results[0]['score']:.3f})")
            # 已删除内容预览日志
            
            # 同时从RAG检索相关原文作为补充
            logger.info("同时从RAG检索相关原文作为补充...")
            search_results = self.rag_system.search(
                query=query,
                top_k=3,
                use_background=True,
                use_hybrid=True
            )
            
            # 构建补充原文
            supplementary_parts = []
            supplementary_sources = []
            
            if search_results['target_source'] == "影化成殇":
                for result in search_results['sequel_results'][:3]:
                    content = result['content']
                    chapter = result['metadata'].get('chapter', '未知章节')
                    supplementary_parts.append(f"[来源: {chapter}]\n{content}")
                    supplementary_sources.append({
                        "source": "影化成殇",
                        "chapter": chapter,
                        "content": content[:200]
                    })
            else:
                for result in search_results['main_results'][:3]:
                    content = result['content']
                    chapter = result['metadata'].get('chapter', '未知章节')
                    supplementary_parts.append(f"[来源: {chapter}]\n{content}")
                    supplementary_sources.append({
                        "source": "雪落成诗",
                        "chapter": chapter,
                        "content": content[:200]
                    })
            
            # 元文本结果作为权威上下文，置顶
            full_context = f"[作者独家解读]\n{combined_content}"
            if supplementary_parts:
                full_context += "\n\n---\n\n" + "\n\n---\n\n".join(supplementary_parts)
            
            # 元文本知识库是结构化数据源，不在UI中显示
            all_sources = [{
                "source": "元文本知识库",
                "chapter": "作者解读",
                "content": combined_content[:500] + "..." if len(combined_content) > 500 else combined_content,
                "hide_in_ui": True  # 标记为隐藏，不在UI中显示
            }] + supplementary_sources  # supplementary_sources是原文，应该显示
            
            return {
                'context': full_context,
                'sources': all_sources,
                'score': top_results[0]['score'],
                'fallback': False
            }
        else:
            logger.warning("元文本知识库未找到结果，需要回退...")
            return {
                'context': '',
                'sources': [],
                'score': 0.0,
                'fallback': True
            }
    
    def _meta_sections_for_query(self, query: str) -> List[str]:
        """
        根据问题意图返回元文本知识库中应优先命中的顶层段落，便于相关问题优先调用结构化信息。
        顶层键：meta_analysis, reader_response_analysis, transmedia_narrative_elements,
        psychological_realism_and_narrative_credibility, linguistic_hybridity_and_translation,
        author_ai_collaboration_metanarrative, detailed_analysis
        """
        q = query.strip().lower()
        preferred = []
        if any(k in q for k in ['音乐', '背景音乐', '配乐', '歌曲', '选曲', '为什么选']):
            preferred.append('transmedia_narrative_elements')
        if any(k in q for k in ['创作意图', '作者意图', '伏笔', '象征', '隐喻', '为什么这样写', '元叙事', '笔者注']):
            preferred.append('meta_analysis')
        if any(k in q for k in ['读者', '解读', '误读', '理解']):
            preferred.append('reader_response_analysis')
        if any(k in q for k in ['虚构', '真实', '二八', '可信', '信任']):
            preferred.append('psychological_realism_and_narrative_credibility')
        if any(k in q for k in ['英文', '语言', '翻译', '古典', '诗词']):
            preferred.append('linguistic_hybridity_and_translation')
        if any(k in q for k in ['ai', '作者与ai', '协作', '合作']):
            preferred.append('author_ai_collaboration_metanarrative')
        if any(k in q for k in ['作者', '创作']):
            preferred.append('detailed_analysis')
        return list(dict.fromkeys(preferred))
    
    def _extract_meta_keywords(self, text: str) -> List[str]:
        """提取元文本检索的关键词"""
        text_clean = re.sub(r'[^\w\s]', '', text.replace('？', '').replace('?', ''))
        keywords = []
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', text_clean)
        stopwords = ['什么', '怎么', '为什么', '哪里', '是谁', '多少', '如何', '这个', '那个']
        keywords = [w for w in words if len(w) >= 2 and w not in stopwords]
        meta_keywords = ['创作', '意图', '背景音乐', '配乐', '伏笔', '象征', '隐喻', '作者', '解读', '元叙事']
        for kw in meta_keywords:
            if kw in text:
                keywords.append(kw)
        return list(set(keywords))
    
    def _format_meta_result(self, key: str, value: any, path: str) -> str:
        """格式化元文本检索结果（避免输出JSON字段名）"""
        result = f"【{key}】\n"
        if isinstance(value, str):
            result += value
        elif isinstance(value, list):
            for i, item in enumerate(value, 1):
                if isinstance(item, str):
                    result += f"{i}. {item}\n"
                elif isinstance(item, dict):
                    # 递归格式化字典，避免JSON格式
                    result += f"{i}. {self._format_dict_naturally(item)}\n"
        elif isinstance(value, dict):
            # 格式化字典为自然语言，避免JSON格式
            formatted = self._format_dict_naturally(value)
            if len(formatted) > 500:
                result += formatted[:500] + "..."
            else:
                result += formatted
        else:
            result += str(value)
        return result
    
    def _format_dict_naturally(self, data: Dict, indent: int = 0) -> str:
        """将字典格式化为自然语言，避免JSON字段名"""
        result = ""
        indent_str = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, str):
                result += f"{indent_str}{value}\n"
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        result += f"{indent_str}• {item}\n"
                    elif isinstance(item, dict):
                        result += f"{indent_str}• {self._format_dict_naturally(item, indent + 1)}"
            elif isinstance(value, dict):
                result += f"{indent_str}{self._format_dict_naturally(value, indent + 1)}"
            else:
                result += f"{indent_str}{str(value)}\n"
        
        return result
