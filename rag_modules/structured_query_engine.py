"""
结构化查询引擎，避免全文传递JSON
根据查询词精准提取相关字段，而不是传递整个知识库
"""
import json
import logging
from typing import Dict, List, Optional, Any
import re

logger = logging.getLogger(__name__)


class StructuredQueryEngine:
    """结构化查询引擎，避免全文传递JSON"""
    
    def __init__(self):
        """初始化查询引擎"""
        pass
    
    def query_index(self, index_type: str, query_terms: str, knowledge_base: Any, max_results: int = 3) -> str:
        """
        基于JSON结构进行精准查询
        
        Args:
            index_type: 索引类型 ("timeline", "psychology", "imagery", "meta")
            query_terms: 查询词
            knowledge_base: 知识库数据
            max_results: 最大返回结果数
        
        Returns:
            格式化后的查询结果字符串
        """
        if not knowledge_base:
            return ""
        
        mapping = {
            "timeline": {
                "fields": ["timepoint", "events"],
                "query_fn": self.query_timeline
            },
            "psychology": {
                "fields": ["名称", "核心定义", "关键识别特征", "相关概念"],
                "query_fn": self.query_psychology_concepts
            },
            "imagery": {
                "fields": ["元素", "文学功能", "上下文锚点"],
                "query_fn": self.query_imagery_system
            },
            "meta": {
                "fields": ["theme", "author_intent", "structural_elements"],
                "query_fn": self.query_meta_analysis
            }
        }
        
        if index_type not in mapping:
            logger.warning(f"未知的索引类型: {index_type}")
            return ""
        
        # 执行结构化查询
        results = mapping[index_type]["query_fn"](query_terms, knowledge_base, max_results)
        
        # 格式化结果
        return self.format_results(index_type, results)
    
    def query_timeline(self, query_terms: str, timeline_data: List[Dict], max_results: int) -> List[Dict]:
        """时间线索引查询优化"""
        if not timeline_data:
            return []
        
        results = []
        query_lower = query_terms.lower()
        
        for item in timeline_data:
            timepoint = item.get("timepoint", "")
            events = item.get("events", [])
            
            # 计算相关性分数
            score = 0
            timepoint_lower = timepoint.lower()
            
            # 检查时间点是否匹配
            if any(term in timepoint_lower for term in query_lower.split() if len(term) > 1):
                score += 2
            
            # 检查事件是否匹配
            matching_events = []
            for event in events:
                event_lower = event.lower()
                if any(term in event_lower for term in query_lower.split() if len(term) > 1):
                    matching_events.append(event)
                    score += 1
            
            if score > 0:
                # 只提取相关字段
                extracted = {
                    "timepoint": timepoint,
                    "events": matching_events if matching_events else events[:3],  # 如果没匹配到，至少返回前3个
                    "relevance_score": score
                }
                results.append(extracted)
        
        # 按相关性排序
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]
    
    def query_psychology_concepts(self, query_terms: str, psychology_data: Dict, max_results: int) -> List[Dict]:
        """心理学概念库查询优化"""
        if not psychology_data or "心理学概念" not in psychology_data:
            return []
        
        concepts = psychology_data.get("心理学概念", [])
        results = []
        query_lower = query_terms.lower()
        
        for concept in concepts:
            name = concept.get("名称", "")
            definition = concept.get("核心定义", "")
            features = concept.get("关键识别特征", [])
            related = concept.get("相关概念", [])
            
            # 计算相关性分数
            score = 0
            
            # 检查名称匹配
            if name.lower() in query_lower or any(term in name.lower() for term in query_lower.split() if len(term) > 1):
                score += 3
            
            # 检查定义匹配
            if any(term in definition.lower() for term in query_lower.split() if len(term) > 1):
                score += 2
            
            # 检查特征匹配
            for feature in features:
                if any(term in str(feature).lower() for term in query_lower.split() if len(term) > 1):
                    score += 1
                    break
            
            # 检查相关概念匹配
            for rel in related:
                if any(term in str(rel).lower() for term in query_lower.split() if len(term) > 1):
                    score += 1
                    break
            
            if score > 0:
                # 只提取相关字段
                extracted = {
                    "名称": name,
                    "核心定义": definition,
                    "关键识别特征": features[:3] if len(features) > 3 else features,  # 最多3个特征
                    "相关概念": related,
                    "relevance_score": score
                }
                results.append(extracted)
        
        # 按相关性排序
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]
    
    def query_imagery_system(self, query_terms: str, imagery_data: List[Dict], max_results: int) -> List[Dict]:
        """意象系统查询优化"""
        if not imagery_data or not isinstance(imagery_data, list):
            return []
        
        results = []
        query_lower = query_terms.lower()
        
        for item in imagery_data:
            element = item.get("元素", "")
            basic_info = item.get("基本信息", {})
            literary_func = item.get("文学功能", {})
            anchors = item.get("上下文锚点", [])
            
            # 计算相关性分数
            score = 0
            
            # 检查元素名称匹配
            if element.lower() in query_lower or any(term in element.lower() for term in query_lower.split() if len(term) > 1):
                score += 3
            
            # 检查象征意义匹配
            symbol_meaning = literary_func.get("象征意义", "")
            if any(term in symbol_meaning.lower() for term in query_lower.split() if len(term) > 1):
                score += 2
            
            # 检查主题关联匹配
            themes = literary_func.get("主题关联", [])
            for theme in themes:
                if any(term in str(theme).lower() for term in query_lower.split() if len(term) > 1):
                    score += 1
                    break
            
            if score > 0:
                # 只提取相关字段
                extracted = {
                    "元素": element,
                    "基本信息": {
                        "类型": basic_info.get("类型", ""),
                        "首次出现": basic_info.get("首次出现", "")
                    },
                    "文学功能": {
                        "象征意义": symbol_meaning,
                        "主题关联": themes[:3] if len(themes) > 3 else themes
                    },
                    "上下文锚点": anchors[:2] if len(anchors) > 2 else anchors,  # 最多2个锚点
                    "relevance_score": score
                }
                results.append(extracted)
        
        # 按相关性排序
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]
    
    def query_meta_analysis(self, query_terms: str, meta_data: Dict, max_results: int) -> List[Dict]:
        """元文本分析库查询优化"""
        if not meta_data:
            return []
        
        results = []
        query_lower = query_terms.lower()
        
        # 递归搜索元文本知识库
        def search_recursive(data: Dict, path: str = "", depth: int = 0) -> List[Dict]:
            """递归搜索JSON结构"""
            if depth > 5:  # 限制递归深度
                return []
            
            found_items = []
            
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    key_lower = key.lower()
                    
                    # 检查键名是否匹配
                    score = 0
                    if any(term in key_lower for term in query_lower.split() if len(term) > 1):
                        score += 2
                    
                    # 检查值是否匹配
                    if isinstance(value, str):
                        if any(term in value.lower() for term in query_lower.split() if len(term) > 1):
                            score += 1
                            found_items.append({
                                "path": current_path,
                                "key": key,
                                "value": value[:500],  # 限制长度
                                "relevance_score": score
                            })
                    elif isinstance(value, (dict, list)):
                        # 递归搜索
                        found_items.extend(search_recursive(value, current_path, depth + 1))
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    found_items.extend(search_recursive(item, f"{path}[{i}]", depth + 1))
            
            return found_items
        
        found_items = search_recursive(meta_data)
        
        # 按分数排序并去重
        seen_paths = set()
        unique_items = []
        for item in sorted(found_items, key=lambda x: x["relevance_score"], reverse=True):
            path = item["path"]
            if path not in seen_paths:
                seen_paths.add(path)
                unique_items.append(item)
                if len(unique_items) >= max_results:
                    break
        
        return unique_items
    
    def format_results(self, index_type: str, results: List[Dict]) -> str:
        """格式化查询结果"""
        if not results:
            return ""
        
        formatted_parts = []
        
        if index_type == "timeline":
            for i, result in enumerate(results, 1):
                timepoint = result.get("timepoint", "")
                events = result.get("events", [])
                formatted_parts.append(f"[时间点: {timepoint}]")
                for event in events:
                    formatted_parts.append(f"- {event}")
                if i < len(results):
                    formatted_parts.append("")
        
        elif index_type == "psychology":
            for i, result in enumerate(results, 1):
                name = result.get("名称", "")
                definition = result.get("核心定义", "")
                features = result.get("关键识别特征", [])
                related = result.get("相关概念", [])
                
                formatted_parts.append(f"【{name}】")
                formatted_parts.append(f"定义: {definition}")
                if features:
                    formatted_parts.append("关键特征:")
                    for feature in features:
                        formatted_parts.append(f"  - {feature}")
                if related:
                    formatted_parts.append(f"相关概念: {', '.join(related)}")
                if i < len(results):
                    formatted_parts.append("")
        
        elif index_type == "imagery":
            for i, result in enumerate(results, 1):
                element = result.get("元素", "")
                basic_info = result.get("基本信息", {})
                literary_func = result.get("文学功能", {})
                anchors = result.get("上下文锚点", [])
                
                formatted_parts.append(f"【{element}】")
                if basic_info.get("首次出现"):
                    formatted_parts.append(f"首次出现: {basic_info['首次出现']}")
                symbol_meaning = literary_func.get("象征意义", "")
                if symbol_meaning:
                    formatted_parts.append(f"象征意义: {symbol_meaning}")
                themes = literary_func.get("主题关联", [])
                if themes:
                    formatted_parts.append(f"主题关联: {', '.join(themes)}")
                if anchors:
                    formatted_parts.append("关键场景:")
                    for anchor in anchors[:2]:  # 最多2个
                        chapter = anchor.get("章节", "")
                        scene = anchor.get("场景", "")
                        if chapter and scene:
                            formatted_parts.append(f"  - {chapter}: {scene}")
                if i < len(results):
                    formatted_parts.append("")
        
        elif index_type == "meta":
            for i, result in enumerate(results, 1):
                key = result.get("key", "")
                value = result.get("value", "")
                path = result.get("path", "")
                
                formatted_parts.append(f"【{key}】")
                formatted_parts.append(f"{value}")
                if i < len(results):
                    formatted_parts.append("")
        
        return "\n".join(formatted_parts)
