"""
统一行为分析模块
根据问题类型（行为解释类/情节推演类）进行行为分析
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from .base_retriever import BaseRetriever
from .structured_query_engine import StructuredQueryEngine

logger = logging.getLogger(__name__)

# 模块级缓存，避免重复加载元文本知识库
_meta_knowledge_cache = None


class UnifiedBehaviorAnalyzer(BaseRetriever):
    """统一行为分析模块"""
    
    def __init__(self, rag_system):
        """
        初始化行为分析模块
        
        Args:
            rag_system: RAG系统实例
        """
        super().__init__(rag_system)
        
        # 数据源路径（与 RAG 系统一致，支持 DATA_SOURCE_DIR 环境变量）
        self.data_dir = getattr(rag_system, "data_dir", "数据源")
        
        # 加载数据源
        self.character_db = self._load_character_db()
        self.timeline_index = self._load_timeline_index()
        self.meta_knowledge = self._load_meta_knowledge()
        self.imagery_system = self._load_imagery_system()
        self.psychology_concepts = self._load_psychology_concepts()
        self.scenario_rules = self._load_scenario_rules()
        
        # 初始化结构化查询引擎
        self.query_engine = StructuredQueryEngine()
        
        logger.info("行为分析模块初始化完成")
    
    def _load_character_db(self) -> Optional[Dict]:
        """加载人物画像库"""
        file_path = os.path.join(self.data_dir, "人物.json")
        if not os.path.exists(file_path):
            logger.warning(f"人物画像库文件不存在: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("人物画像系统", {}).get("数据", {})
        except Exception as e:
            logger.error(f"加载人物画像库失败: {e}")
            return None
    
    def _load_timeline_index(self) -> Optional[str]:
        """加载时间线索引"""
        file_path = os.path.join(self.data_dir, "时间线梳理.txt")
        if not os.path.exists(file_path):
            logger.warning(f"时间线索引文件不存在: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"加载时间线索引失败: {e}")
            return None
    
    def _load_meta_knowledge(self) -> Optional[Dict]:
        """加载元文本知识库（支持多个JSON对象，使用模块级缓存避免重复加载）"""
        global _meta_knowledge_cache
        
        # 如果已缓存，直接返回
        if _meta_knowledge_cache is not None:
            logger.debug("使用缓存的元文本知识库")
            return _meta_knowledge_cache
        
        file_path = os.path.join(self.data_dir, "元文本知识库.json")
        if not os.path.exists(file_path):
            logger.warning(f"元文本知识库文件不存在: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析多个JSON对象并合并（与meta_knowledge_retriever.py保持一致）
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
            
            if not objs:
                # 如果没有找到多个对象，尝试直接解析整个文件
                try:
                    result = json.loads(content)
                    # 缓存结果
                    _meta_knowledge_cache = result
                    logger.info("成功加载元文本知识库（单个JSON对象）")
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"无法解析JSON文件: {e}")
                    return None
            
            # 合并所有对象
            merged = {}
            for obj in objs:
                for key, value in obj.items():
                    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key] = self._deep_merge_dict(merged[key], value)
                    else:
                        merged[key] = value
            
            logger.info(f"成功加载元文本知识库，包含 {len(objs)} 个JSON对象")
            # 缓存加载结果
            _meta_knowledge_cache = merged
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
    
    def _load_imagery_system(self) -> Optional[List[Dict]]:
        """加载意象系统"""
        file_path = os.path.join(self.data_dir, "意象系统.txt")
        if not os.path.exists(file_path):
            logger.warning(f"意象系统文件不存在: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 尝试解析为JSON（如果是JSON格式）
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # 如果不是JSON，返回原始文本
                    return content
        except Exception as e:
            logger.error(f"加载意象系统失败: {e}")
            return None
    
    def _load_psychology_concepts(self) -> Optional[Dict]:
        """加载小说心理学概念库（用于行为解释类）"""
        file_path = os.path.join(self.data_dir, "小说心理学概念库.json")
        if not os.path.exists(file_path):
            logger.warning(f"小说心理学概念库文件不存在: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载小说心理学概念库失败: {e}")
            return None
    
    def _load_scenario_rules(self) -> Optional[Dict]:
        """加载情节推演规则（用于情节推演类）"""
        file_path = os.path.join(self.data_dir, "情节推演规则.json")
        if not os.path.exists(file_path):
            logger.warning(f"情节推演规则文件不存在: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载情节推演规则失败: {e}")
            return None
    
    def analyze(self, query: str, question_type: str, ernie_parse_result: Optional[Dict] = None, rag_original_text: Optional[str] = None) -> Dict:
        """
        根据问题类型进行行为分析
        
        Args:
            query: 用户查询
            question_type: 问题类型（behavior_explanation 或 scenario_simulation）
            ernie_parse_result: ERNIE解析结果（可选）
            rag_original_text: RAG检索到的原文片段（用于事实核查）
        
        Returns:
            结构化分析数据
        """
        # 提取分析需求
        if ernie_parse_result:
            requirements = ernie_parse_result.get("analysis_requirements", {})
            entities = ernie_parse_result.get("extracted_entities", {})
        else:
            # 如果没有ERNIE解析结果，从query中简单提取
            requirements = {}
            entities = self._extract_entities_from_query(query)
        
        # 1. 根据需求调取数据
        data_package = self.retrieve_data_package(requirements, entities, query)
        
        # 2. 根据问题类型构建分析框架（框架中已包含元文本知识库）
        if question_type == "scenario_simulation":
            framework = self.build_scenario_framework(data_package, query, entities)
        else:  # behavior_explanation
            framework = self.build_behavior_explanation_framework(data_package, query, entities)
        
        # 3. 转换为LLM友好的结构化数据（传入RAG原文片段和查询）
        structured_data = self.format_for_llm(framework, question_type, rag_original_text, query=query)
        
        return structured_data
    
    def _extract_entities_from_query(self, query: str) -> Dict:
        """从查询中简单提取实体（如果没有ERNIE解析）"""
        entities = {
            "characters": [],
            "key_events": [],
            "time_periods": [],
            "locations": []
        }
        
        # 简单的人物名称识别（可以根据实际需求扩展）
        if self.character_db:
            for char_name in self.character_db.keys():
                if char_name in query:
                    entities["characters"].append(char_name)
        
        return entities
    
    def retrieve_data_package(self, requirements: Dict, entities: Dict, query: str) -> Dict:
        """根据需求调取数据包"""
        data = {}
        
        # 提取人物信息
        characters = entities.get("characters", [])
        if not characters and self.character_db:
            # 如果没有明确指定人物，尝试从查询中识别
            for char_name in self.character_db.keys():
                if char_name in query:
                    characters.append(char_name)
        
        # 人物画像
        if characters and self.character_db:
            data["character_models"] = {}
            for char in characters:
                if char in self.character_db:
                    data["character_models"][char] = self.character_db[char]
        
        # 时间线索引
        if self.timeline_index:
            data["timeline"] = self.timeline_index
        
        # 元文本知识库
        if self.meta_knowledge:
            data["meta_knowledge"] = self.meta_knowledge
        
        # 意象系统
        if self.imagery_system:
            data["imagery"] = self.imagery_system
        
        return data
    
    def build_scenario_framework(self, data_package: Dict, query: str, entities: Dict) -> Dict:
        """构建情节推演框架"""
        framework = {
            "analysis_type": "scenario_simulation",
            "query": query,
            "character_states": {},
            "decision_points": [],
            "environment_constraints": {},
            "possible_paths": []
        }
        
        # 提取人物状态
        character_models = data_package.get("character_models", {})
        for char_name, char_data in character_models.items():
            framework["character_states"][char_name] = {
                "心理模型": char_data.get("心理模型", {}),
                "关键事件": char_data.get("key_events", []),
                "关系": char_data.get("relationships", [])
            }
        
        # 提取时间线信息
        if data_package.get("timeline"):
            framework["timeline_context"] = data_package["timeline"]
        
        # 提取元文本知识库
        if data_package.get("meta_knowledge"):
            framework["meta_knowledge"] = data_package["meta_knowledge"]
        
        # 提取假设性变化
        if "如果" in query or "假设" in query or "要是" in query:
            framework["hypothetical_change"] = query
        
        return framework
    
    def build_behavior_explanation_framework(self, data_package: Dict, query: str, entities: Dict) -> Dict:
        """构建行为解释框架"""
        framework = {
            "analysis_type": "behavior_explanation",
            "query": query,
            "character_models": {},
            "behavior_instances": [],
            "psychology_patterns": [],
            "development_over_time": {}
        }
        
        # 提取人物心理模型
        character_models = data_package.get("character_models", {})
        for char_name, char_data in character_models.items():
            framework["character_models"][char_name] = {
                "心理模型": char_data.get("心理模型", {}),
                "关键事件": char_data.get("key_events", []),
                "行为模式": char_data.get("心理模型", {}).get("行为模式", [])
            }
        
        # 提取时间线信息
        if data_package.get("timeline"):
            framework["timeline_context"] = data_package["timeline"]
        
        # 提取元文本知识库
        if data_package.get("meta_knowledge"):
            framework["meta_knowledge"] = data_package["meta_knowledge"]
        
        # 提取意象信息（如果相关）
        if data_package.get("imagery"):
            framework["imagery_context"] = data_package["imagery"]
        
        return framework
    
    def format_for_llm(self, framework: Dict, question_type: str, rag_original_text: Optional[str] = None, query: str = "") -> Dict:
        """
        转换为LLM友好的结构化数据
        按照事实权威性排序：时间线索引 > RAG检索原文 > 元文本分析库 > 人物心理模型 > 动态摘要
        """
        structured = {
            "analysis_type": framework.get("analysis_type"),
            "context": "",
            "sources": [],
            "additional_knowledge": None
        }
        
        # 构建上下文文本（按权威性顺序）
        context_parts = []
        
        # 1. 时间线索引（最高权威）- 使用结构化查询
        if framework.get("timeline_context"):
            timeline_data = framework["timeline_context"]
            if isinstance(timeline_data, list):
                # 使用结构化查询引擎
                timeline_result = self.query_engine.query_index(
                    "timeline", query, timeline_data, max_results=3
                )
                if timeline_result:
                    context_parts.append("【1. 时间线索引】（最高权威事实）")
                    context_parts.append(timeline_result)
            else:
                # 兼容旧格式
                context_parts.append("【1. 时间线索引】（最高权威事实）")
                context_parts.append(str(timeline_data)[:2000])
        
        # 2. RAG检索到的原文片段（用于事实核查）
        if rag_original_text:
            context_parts.append("\n【2. RAG检索到的原文片段】（用于事实核查）")
            context_parts.append(rag_original_text[:3000])  # 限制长度
        
        # 3. 元文本分析库（作者提供的创作说明）- 使用结构化查询
        if framework.get("meta_knowledge"):
            meta_knowledge = framework["meta_knowledge"]
            if isinstance(meta_knowledge, dict):
                # 使用结构化查询引擎
                meta_result = self.query_engine.query_index(
                    "meta", query, meta_knowledge, max_results=3
                )
                if meta_result:
                    context_parts.append("\n【3. 元文本分析库】（作者创作说明）")
                    context_parts.append(meta_result)
        
        # 4. 人物心理模型（基于文本推断的稳定特征）
        if question_type == "scenario_simulation":
            if framework.get("character_states"):
                context_parts.append("\n【4. 相关人物心理模型】（基于文本推断）")
                for char_name, char_state in framework["character_states"].items():
                    context_parts.append(f"\n## {char_name}")
                    context_parts.append(json.dumps(char_state, ensure_ascii=False, indent=2))
        else:  # behavior_explanation
            if framework.get("character_models"):
                context_parts.append("\n【4. 相关人物心理模型】（基于文本推断）")
                for char_name, char_model in framework["character_models"].items():
                    context_parts.append(f"\n## {char_name}")
                    context_parts.append(json.dumps(char_model, ensure_ascii=False, indent=2))
        
        # 5. 专业分析工具（心理学概念库或情节推演规则）
        if question_type == "scenario_simulation":
            if self.scenario_rules:
                structured["additional_knowledge"] = {
                    "type": "scenario_rules",
                    "content": self.scenario_rules
                }
                context_parts.append("\n【5. 情节推演规则】")
                context_parts.append(json.dumps(self.scenario_rules, ensure_ascii=False, indent=2))
            structured["sources"].append({
                "source": "情节推演规则库",
                "type": "scenario_rules"
            })
        else:  # behavior_explanation
            if self.psychology_concepts:
                # 使用结构化查询引擎
                psychology_result = self.query_engine.query_index(
                    "psychology", query, self.psychology_concepts, max_results=3
                )
                if psychology_result:
                    structured["additional_knowledge"] = {
                        "type": "psychology_concepts",
                        "content": psychology_result
                    }
                    context_parts.append("\n【5. 小说心理学概念库】")
                    context_parts.append(psychology_result)
            structured["sources"].append({
                "source": "小说心理学概念库",
                "type": "psychology_concepts"
            })
        
        # 6. 意象系统（如果相关，作为补充）- 使用结构化查询
        if framework.get("imagery_context"):
            imagery_data = framework["imagery_context"]
            if isinstance(imagery_data, list):
                # 使用结构化查询引擎
                imagery_result = self.query_engine.query_index(
                    "imagery", query, imagery_data, max_results=2
                )
                if imagery_result:
                    context_parts.append("\n【6. 意象系统】（补充信息）")
                    context_parts.append(imagery_result)
            else:
                # 兼容旧格式
                context_parts.append("\n【6. 意象系统】（补充信息）")
                context_parts.append(str(imagery_data)[:1000])
        
        structured["context"] = "\n".join(context_parts)
        
        return structured
    
    def retrieve(self, query: str, question_type: str = "behavior_explanation", top_k: int = 5, rag_original_text: Optional[str] = None) -> Dict:
        """
        检索相关上下文（实现BaseRetriever接口）
        
        Args:
            query: 查询文本
            question_type: 问题类型
            top_k: 返回top k个结果（此模块不使用）
            rag_original_text: RAG检索到的原文片段（用于事实核查）
        
        Returns:
            检索结果字典
        """
        # 调用analyze方法（传入RAG原文片段）
        result = self.analyze(query, question_type, rag_original_text=rag_original_text)
        
        return {
            'context': result.get('context', ''),
            'sources': result.get('sources', []),
            'score': 1.0,  # 行为分析模块的分数固定为1.0
            'fallback': False,
            'additional_knowledge': result.get('additional_knowledge'),
            'analysis_type': result.get('analysis_type')
        }
