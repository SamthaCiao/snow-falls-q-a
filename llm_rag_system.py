"""
LLM + RAG 智能系统
使用LLM判断是否需要调用RAG，并综合输出结果
支持三重索引和混合检索
"""
import os
import json
import logging
import requests
import time
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from rag_system import NovelRAGSystem
from rag_modules import (
    GlobalSummaryRetriever,
    TimelineRetriever,
    CharacterArcRetriever,
    ImageryRetriever,
    MetaKnowledgeRetriever,
    ChapterRAGRetriever,
    ChapterSummaryRetriever,
    DynamicSummaryGenerator,
    UnifiedBehaviorAnalyzer,
    ReasoningModule
)
try:
    from rag_modules.langchain_adapters import (
        build_langchain_retriever_map,
        documents_to_result,
        HAS_LANGCHAIN as _HAS_LC_ADAPTER,
    )
except ImportError:
    build_langchain_retriever_map = None  # type: ignore
    documents_to_result = None  # type: ignore
    _HAS_LC_ADAPTER = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到终端
    ]
)
logger = logging.getLogger(__name__)


class LLMRAGSystem:
    """LLM + RAG 智能系统"""
    
    def __init__(self, 
                 rag_system: Optional[NovelRAGSystem] = None,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 model: str = "gpt-3.5-turbo"):
        """
        初始化LLM+RAG系统
        
        Args:
            rag_system: RAG系统实例，如果为None则自动创建
            api_key: OpenAI API密钥，如果为None则从环境变量读取
            api_base: API基础URL，如果为None则使用OpenAI默认
            model: 使用的模型名称
        """
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("开始初始化LLM+RAG系统...")
        logger.info("=" * 80)
        
        # 步骤1: 初始化RAG系统
        step_start = time.time()
        if rag_system is None:
            logger.info("[步骤1/7] 正在初始化RAG系统...")
            data_dir = os.environ.get("DATA_SOURCE_DIR") or "数据源"
            self.rag_system = NovelRAGSystem(
                data_dir=data_dir,
                chunk_size=400,
                chunk_overlap=50
            )
            # 构建索引(如果不存在)
            logger.info("        正在构建索引(如果不存在)...")
            self.rag_system.build_index(force_rebuild=False)
        else:
            logger.info("[步骤1/7] 使用已提供的RAG系统实例")
            self.rag_system = rag_system
        step_time = time.time() - step_start
        logger.info(f"        ✓ RAG系统初始化完成 (耗时: {step_time:.2f}秒)")
        
        # 步骤2: 加载环境变量
        step_start = time.time()
        logger.info("[步骤2/7] 正在加载环境变量...")
        load_dotenv('SUPER_MIND_API_KEY.env')
        load_dotenv('.env')
        step_time = time.time() - step_start
        logger.info(f"        ✓ 环境变量加载完成 (耗时: {step_time:.2f}秒)")
        
        # 步骤3: 读取API密钥
        step_start = time.time()
        logger.info("[步骤3/7] 正在读取API密钥...")
        api_key = api_key or os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY')
        api_base = api_base or os.getenv('OPENAI_API_BASE')
        
        if not api_key:
            raise ValueError(
                "未找到API密钥！\n"
                "请在以下位置之一设置API密钥：\n"
                "1. SUPER_MIND_API_KEY.env 文件中设置 DEEPSEEK_API_KEY\n"
                "2. .env 文件中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY\n"
                "3. 环境变量中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY"
            )
        
        # 判断是否使用DeepSeek API
        is_deepseek = bool(os.getenv('DEEPSEEK_API_KEY')) or (api_key and 'deepseek' in api_key.lower())
        step_time = time.time() - step_start
        logger.info(f"        ✓ API密钥读取完成 (耗时: {step_time:.2f}秒)")
        logger.info(f"        使用API类型: {'DeepSeek' if is_deepseek else 'OpenAI'}")
        
        # 步骤4: 初始化OpenAI客户端
        step_start = time.time()
        logger.info("[步骤4/7] 正在初始化OpenAI客户端...")
        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        else:
            # 如果使用DeepSeek但没有设置base_url，使用DeepSeek的默认地址
            if is_deepseek:
                client_kwargs["base_url"] = "https://api.deepseek.com/v1"
        
        self.client = OpenAI(**client_kwargs)
        step_time = time.time() - step_start
        logger.info(f"        ✓ OpenAI客户端初始化完成 (耗时: {step_time:.2f}秒)")
        
        # 步骤5: 选择模型
        step_start = time.time()
        logger.info("[步骤5/7] 正在选择模型...")
        # 根据API类型自动选择模型
        if model == "gpt-3.5-turbo":  # 如果使用默认模型
            if is_deepseek:
                # DeepSeek的模型名称
                self.model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
            else:
                self.model = model
        else:
            self.model = model
        step_time = time.time() - step_start
        logger.info(f"        ✓ 模型选择完成: {self.model} (耗时: {step_time:.2f}秒)")
        
        # 步骤6: 检查百度API配置
        step_start = time.time()
        logger.info("[步骤6/7] 正在检查百度大模型API配置...")
        self.baidu_api_key = os.getenv('BAIDU_API_KEY')
        if self.baidu_api_key:
            logger.info("        ✓ 百度大模型API已配置，将用于问题类型判断")
        else:
            logger.warning("        ⚠ 未配置百度大模型API，将使用DeepSeek进行问题类型判断")
        step_time = time.time() - step_start
        logger.info(f"        ✓ API配置检查完成 (耗时: {step_time:.2f}秒)")
        
        # 步骤7: 初始化RAG检索器模块
        step_start = time.time()
        logger.info("[步骤7/8] 正在初始化RAG检索器模块...")
        
        # 检查动态摘要功能（默认启用，可通过环境变量禁用）
        self.enable_dynamic_summary = os.getenv('ENABLE_DYNAMIC_SUMMARY', 'true').lower() == 'true'
        if self.enable_dynamic_summary:
            logger.info("        动态摘要功能: 已启用")
        else:
            logger.info("        动态摘要功能: 未启用")
        
        # 初始化各个RAG检索器模块
        logger.info("        正在加载检索器模块...")
        self.retrievers = {
            "global_summary": GlobalSummaryRetriever(self.rag_system),
            "timeline_db": TimelineRetriever(self.rag_system),
            "character_arc": CharacterArcRetriever(self.rag_system),
            "imagery": ImageryRetriever(self.rag_system),
            "meta_analysis": MetaKnowledgeRetriever(self.rag_system),
            "chapter_rag": ChapterRAGRetriever(self.rag_system),
            "chapter_summary": ChapterSummaryRetriever(self.rag_system),
            "behavior_explanation": UnifiedBehaviorAnalyzer(self.rag_system),
            "scenario_simulation": UnifiedBehaviorAnalyzer(self.rag_system)
        }
        logger.info(f"        ✓ 已加载 {len(self.retrievers)} 个检索器模块")
        
        # LangChain 编排：将现有检索器封装为 BaseRetriever，便于链式编排与可观测性
        self._lc_retriever_map: Dict[str, Any] = {}
        if build_langchain_retriever_map and _HAS_LC_ADAPTER:
            try:
                self._lc_retriever_map = build_langchain_retriever_map(self.retrievers)
                logger.info(f"        ✓ LangChain 检索器映射已构建 ({len(self._lc_retriever_map)} 个)")
            except Exception as e:
                logger.warning(f"        ⚠ LangChain 适配层构建失败: {e}，将使用原有检索调用")
        
        # 初始化动态摘要生成器
        if self.enable_dynamic_summary:
            logger.info("        正在初始化动态摘要生成器...")
            self.dynamic_summary_generator = DynamicSummaryGenerator(
                self.rag_system, self.client, self.model
            )
            logger.info("        ✓ 动态摘要生成器初始化完成")
        else:
            self.dynamic_summary_generator = None
        
        step_time = time.time() - step_start
        logger.info(f"        ✓ RAG检索器模块初始化完成 (耗时: {step_time:.2f}秒)")
        
        # 步骤8: 初始化Reasoning模块
        step_start = time.time()
        logger.info("[步骤8/8] 正在初始化Reasoning模块...")
        try:
            # 从环境变量读取超时时间，默认120秒（thinking mode需要更长时间）
            reasoning_timeout = int(os.getenv('REASONING_TIMEOUT', '120'))
            self.reasoning_module = ReasoningModule(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                timeout=reasoning_timeout
            )
            logger.info(f"        ✓ Reasoning模块初始化完成（超时时间: {reasoning_timeout}秒）")
        except Exception as e:
            logger.warning(f"        ⚠ Reasoning模块初始化失败: {e}，将禁用多跳检索功能")
            self.reasoning_module = None
        step_time = time.time() - step_start
        logger.info(f"        ✓ Reasoning模块初始化完成 (耗时: {step_time:.2f}秒)")
        
        # 总结
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"✓ LLM+RAG系统初始化完成！")
        logger.info(f"  总耗时: {total_time:.2f}秒")
        logger.info(f"  使用模型: {self.model}")
        logger.info(f"  检索器数量: {len(self.retrievers)}")
        logger.info(f"  动态摘要: {'启用' if self.enable_dynamic_summary else '禁用'}")
        logger.info(f"  多跳检索: {'启用' if self.reasoning_module else '禁用'}")
        logger.info("=" * 80)
    
    def _should_use_dynamic_summary(self, query: str, question_type: str) -> Tuple[bool, str]:
        """
        判断是否需要使用动态摘要
        
        Args:
            query: 用户查询
            question_type: 问题类型
        
        Returns:
            (是否需要动态摘要, 判断理由)
        """
        # 如果动态摘要功能未启用，直接返回False
        if not self.enable_dynamic_summary:
            return False, "动态摘要功能未启用"
        
        # 复杂问题特征判断
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
        
        # 3. 预设摘要匹配度判断（如果问题类型是global_summary但查询很具体）
        if question_type == "global_summary":
            # 如果问题很具体，可能需要动态摘要
            specific_keywords = ["为什么", "如何", "怎样", "原因", "过程", "细节"]
            if any(kw in query_lower for kw in specific_keywords) and len(query) > 15:
                return True, "全局摘要无法覆盖的具体问题"
        
        # 4. 需要深度推理的问题
        reasoning_keywords = ["为什么", "原因", "导致", "影响", "结果", "后果", "意义"]
        if any(kw in query_lower for kw in reasoning_keywords) and len(query) > 25:
            return True, "需要深度推理的问题"
        
        return False, "简单问题，使用预设索引"
    
    def _generate_dynamic_summary(self, query: str, retrieved_chapters: List[Dict]) -> Tuple[str, bool]:
        """
        基于检索到的章节内容，生成针对问题的动态摘要
        
        Args:
            query: 用户查询
            retrieved_chapters: 检索到的章节内容列表，每个元素包含content和metadata
        
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
        max_length = 4000  # 限制总长度，避免超出模型上下文
        
        for chapter, contents in chapters_content.items():
            chapter_text = f"\n## {chapter}\n" + "\n\n".join(contents[:3])  # 每个章节最多3个片段
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文本摘要专家，擅长从长文本中提取关键信息并生成聚焦的摘要。"},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            summary = response.choices[0].message.content.strip()
            
            # 验证摘要质量（简单检查：长度、是否包含关键信息）
            if len(summary) < 100:
                logger.warning("动态摘要过短，质量可能不足")
                return summary, False
            
            # 检查是否包含查询中的关键词
            query_keywords = [w for w in query.split() if len(w) > 1]
            matched_keywords = sum(1 for kw in query_keywords if kw.lower() in summary.lower())
            if matched_keywords < len(query_keywords) * 0.3:  # 至少匹配30%的关键词
                logger.warning("动态摘要可能未充分覆盖查询关键词")
                return summary, False
            
            logger.info(f"动态摘要生成成功，长度: {len(summary)} 字符")
            return summary, True
            
        except Exception as e:
            logger.error(f"动态摘要生成失败: {e}")
            return "", False
    
    def _question_classifier(self, question: str) -> str:
        """
        判断问题类型，选择最合适的索引策略
        
        Args:
            question: 用户问题
        
        Returns:
            问题类型字符串：
            - "global_summary": 使用全局概要
            - "timeline_db": 使用时间线索引
            - "character_arc": 使用人物关系索引（暂时映射到章节RAG）
            - "chapter_rag": 使用现有RAG
        """
        # 宏观总结类问题
        macro_keywords = ["主要剧情", "故事梗概", "整体内容", "全书讲了什么", "情节发展", 
                         "讲了什么", "主要内容", "主题", "概括", "总结", "整体", 
                         "第一部讲了什么", "第二部讲了什么", "主要讲了什么"]
        
        # 时间线类问题  
        timeline_keywords = ["什么时候", "时间顺序", "先后", "发展到哪一步", "时间线", 
                           "时间顺序", "时间发展", "时间进程", "时间脉络"]
        
        # 人物关系问题
        character_keywords = ["关系发展", "感情线", "如何走到一起", "关系变化", 
                             "感情发展", "关系演变"]
        
        # 意象类问题
        imagery_keywords = ["意象", "象征", "隐喻", "比喻", "符号", "意义", "栀子花", "含羞草", 
                           "冰锐", "法拉第笼", "樱花", "星月耳坠", "黑板", "别墅", "行宫", 
                           "曼康基猫", "沙漠", "沙漏", "钢琴", "琴声", "牛皮糖", "星空", 
                           "星河", "星火", "天台", "影", "殇", "雪", "诗", "肖申克的救赎", 
                           "伞", "跑步", "来生酒吧", "意象系统", "文学功能", "象征意义", 
                           "情感演变", "主题关联", "上下文锚点"]
        
        # 创作意图类问题（新增）
        meta_analysis_keywords = [
            # 直接意图
            "创作意图", "作者想表达", "为什么这样写", "隐喻", "作者意图", "创作目的",
            # 背景知识
            "背景音乐", "灵感来源", "场景参考", "配乐", "音乐选择", "为什么选这首歌",
            # 深层分析
            "伏笔", "象征", "结局考量", "为什么这样设计", "作者注", "笔者注",
            "元叙事", "叙事策略", "创作技巧", "为什么用", "为什么是"
        ]
        
        # 事实类关键词（需结合时间线/事件链条做核查，走 chapter_rag 以同时获得 RAG+时间线+事件链条）
        fact_keywords = ["发生在", "第一次", "哪一章", "是谁", "做了什么", "出现在", "在哪里", "哪一集", "具体事件", "第几章", "什么时候发生的"]
        
        # 判断逻辑
        question_lower = question.lower()
        
        # 优先判断创作意图类问题
        if any(kw in question_lower for kw in meta_analysis_keywords):
            return "meta_analysis"  # 使用元文本知识库
        elif any(kw in question_lower for kw in macro_keywords):
            return "global_summary"  # 使用全局概要
        # 事实+时间类：走 chapter_rag，以便同时使用时间线、关键事件链条与 RAG 原文
        elif any(kw in question_lower for kw in fact_keywords) and any(kw in question_lower for kw in timeline_keywords):
            return "chapter_rag"
        elif any(kw in question_lower for kw in timeline_keywords):
            return "timeline_db"  # 使用时间线索引
        elif any(kw in question_lower for kw in character_keywords):
            return "character_arc"  # 使用人物关系索引（暂时映射到章节RAG）
        elif any(kw in question_lower for kw in imagery_keywords):
            return "imagery"  # 使用意象系统索引
        else:
            return "chapter_rag"  # 使用现有RAG
    
    def _classify_question_type(self, user_query: str) -> Tuple[str, str]:
        """
        智能路由判断：根据问题类型选择最合适的索引策略
        
        Returns:
            (问题类型, 判断理由)
            问题类型: 
            - "global_summary" (全局摘要) - 宏观总结类问题
            - "timeline_db" (时间线) - 时间顺序类问题
            - "character_arc" (人物关系) - 人物关系发展类问题
            - "imagery" (意象系统) - 意象、象征、隐喻等概念性问题
            - "chapter_rag" (章节RAG) - 具体章节或事实类问题
        """
        # 使用智能路由判断机制
        question_type = self._question_classifier(user_query)
        
        if not self.baidu_api_key:
            # 如果没有百度API，直接返回关键词判断结果
            type_names = {
                "global_summary": "全局摘要",
                "timeline_db": "时间线索引",
                "character_arc": "人物关系索引",
                "imagery": "意象系统索引",
                "chapter_rag": "章节RAG检索"
            }
            return question_type, f"关键词匹配：{type_names.get(question_type, '章节RAG检索')}"
        
        # 使用百度大模型进行更精确的判断
        try:
            # 百度千帆API调用（新格式）
            url = "https://qianfan.baidubce.com/v2/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.baidu_api_key}"
            }
            
            prompt = f"""请判断以下问题属于哪种类型，选择最合适的索引策略：

问题：{user_query}

类型说明：
1. "global_summary" (全局摘要)：询问故事整体内容、主题、梗概、主要情节等宏观问题
   例如："第一部讲了什么故事"、"这本书的主题是什么"、"整体概括一下"、"主要剧情是什么"

2. "timeline_db" (时间线)：询问时间顺序、发展进程、时间脉络等问题
   例如："什么时候发生的"、"时间顺序是什么"、"发展到哪一步"、"时间线"

3. "character_arc" (人物关系)：询问人物关系发展、感情线演变等问题
   例如："关系如何发展"、"感情线"、"如何走到一起"

4. "imagery" (意象系统)：询问意象、象征、隐喻、符号意义等概念性问题
   例如："栀子花的象征意义"、"冰锐的文学功能"、"法拉第笼代表什么"、"意象系统"

5. "meta_analysis" (元文本分析)：询问创作意图、背景音乐、伏笔、作者解读等元文本问题
   例如："为什么选择这首歌作为背景音乐"、"这个章节的创作意图是什么"、"作者想表达什么"、"为什么这样写"

6. "behavior_explanation" (行为解释类)：询问人物行为解释、心理动机、行为模式等行为分析类问题
   例如："为什么雪遗诗会这样做"、"夏空山的行为动机是什么"、"这种行为模式的心理原因"

7. "scenario_simulation" (情节推演类)：询问假设性情节推演、如果...会怎样等情节推演类问题
   例如："如果雪遗诗没有表白会怎样"、"假设夏空山没有发绝交短信"、"要是他们在一起了会怎样"

8. "chapter_rag" (章节RAG)：询问具体人物、事件、细节等微观问题
   例如："雪遗诗班上的班花是谁"、"印子喜欢的体育运动"、"小蓝老头指的是什么"

只返回JSON格式：{{"type": "global_summary" | "timeline_db" | "character_arc" | "imagery" | "meta_analysis" | "behavior_explanation" | "scenario_simulation" | "chapter_rag", "reason": "判断理由"}}"""

            payload = {
                "model": "ernie-3.5-8k",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
            }
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 检查是否有错误
                if 'error' in result:
                    error_code = result.get('error', {}).get('code', '')
                    error_msg = result.get('error', {}).get('message', '')
                    logger.warning(f"百度API返回错误: code={error_code}, message={error_msg}")
                    raise Exception(f"百度API错误: {error_msg}")
                
                # 新格式的响应中，内容在 choices[0].message.content
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    
                    # 处理可能被包裹在代码块中的JSON
                    content_clean = content.strip()
                    if content_clean.startswith('```json'):
                        # 移除 ```json 和 ``` 标记
                        content_clean = content_clean.replace('```json', '').replace('```', '').strip()
                    elif content_clean.startswith('```'):
                        # 移除通用的 ``` 标记
                        content_clean = content_clean.replace('```', '').strip()
                    
                    try:
                        # 尝试解析JSON
                        parsed = json.loads(content_clean)
                        q_type = parsed.get("type", question_type)  # 如果解析失败，使用关键词判断结果
                        reason = parsed.get("reason", "")
                        
                        # 验证类型是否有效
                        valid_types = ["global_summary", "timeline_db", "character_arc", "imagery", "meta_analysis", "behavior_explanation", "scenario_simulation", "chapter_rag"]
                        if q_type not in valid_types:
                            q_type = question_type  # 使用关键词判断结果
                    except json.JSONDecodeError:
                        # 如果JSON解析失败，使用关键词判断结果
                        q_type = question_type
                        reason = "JSON解析失败，使用关键词判断"
                    
                    logger.info(f"问题类型判断: {q_type} - {reason}")
                    return q_type, reason
                else:
                    logger.warning(f"百度API响应中未找到choices字段: {result}")
                    raise Exception("百度API响应格式异常")
            else:
                logger.warning(f"百度API调用失败，状态码: {response.status_code}, 响应: {response.text}")
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"问题类型判断失败: {e}，使用关键词判断")
            # 如果判断失败，使用关键词判断结果
            type_names = {
                "global_summary": "全局摘要",
                "timeline_db": "时间线索引",
                "character_arc": "人物关系索引",
                "imagery": "意象系统索引",
                "meta_analysis": "元文本分析",
                "behavior_explanation": "行为解释类",
                "scenario_simulation": "情节推演类",
                "chapter_rag": "章节RAG检索"
            }
            return question_type, f"关键词匹配：{type_names.get(question_type, '章节RAG检索')}"
    
    def _load_meta_knowledge(self) -> Optional[Dict]:
        """
        加载元文本知识库
        
        Returns:
            元文本知识库字典，如果加载失败返回None
        """
        data_dir = getattr(self.rag_system, "data_dir", "数据源")
        meta_file = os.path.join(data_dir, "元文本知识库.json")
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
                # 合并列表（去重）
                result[key] = result[key] + [item for item in value if item not in result[key]]
            else:
                result[key] = value
        return result
    
    def _retrieve_meta_knowledge(self, query: str, chapter_id: Optional[str] = None) -> Dict:
        """
        从元文本知识库中检索相关信息
        
        Args:
            query: 用户问题
            chapter_id: 章节ID（如果可以从问题中提取）
        
        Returns:
            包含检索到的元文本信息的字典
        """
        if not self.meta_knowledge:
            return {
                'content': '',
                'sources': [],
                'score': 0.0
            }
        
        results = []
        query_lower = query.lower()
        query_keywords = self._extract_meta_keywords(query)
        
        # 递归搜索元文本知识库
        def search_recursive(data, path="", depth=0):
            """递归搜索JSON结构"""
            if depth > 10:  # 限制递归深度
                return
            
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # 检查键名是否匹配
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
                        # 构建结果文本
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
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 提取前5个最相关的结果
        top_results = results[:5]
        
        if top_results:
            # 合并内容
            combined_content = "\n\n".join([r['content'] for r in top_results])
            return {
                'content': combined_content,
                'sources': [{'path': r['path'], 'score': r['score']} for r in top_results],
                'score': top_results[0]['score'] if top_results else 0.0
            }
        else:
            return {
                'content': '',
                'sources': [],
                'score': 0.0
            }
    
    def _extract_meta_keywords(self, text: str) -> List[str]:
        """提取元文本检索的关键词"""
        import re
        # 移除标点和问号
        text_clean = re.sub(r'[^\w\s]', '', text.replace('？', '').replace('?', ''))
        
        # 提取重要词汇
        keywords = []
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', text_clean)
        
        # 过滤停用词
        stopwords = ['什么', '怎么', '为什么', '哪里', '是谁', '多少', '如何', '这个', '那个']
        keywords = [w for w in words if len(w) >= 2 and w not in stopwords]
        
        # 添加特殊关键词
        meta_keywords = ['创作', '意图', '背景音乐', '配乐', '伏笔', '象征', '隐喻', '作者', '解读', '元叙事']
        for kw in meta_keywords:
            if kw in text:
                keywords.append(kw)
        
        return list(set(keywords))
    
    def _format_meta_result(self, key: str, value: any, path: str) -> str:
        """格式化元文本检索结果"""
        result = f"【{key}】\n"
        
        if isinstance(value, str):
            result += value
        elif isinstance(value, list):
            for i, item in enumerate(value, 1):
                if isinstance(item, str):
                    result += f"{i}. {item}\n"
                elif isinstance(item, dict):
                    result += f"{i}. {json.dumps(item, ensure_ascii=False, indent=2)}\n"
        elif isinstance(value, dict):
            # 只显示前3层，避免过长
            result += json.dumps(value, ensure_ascii=False, indent=2)[:500]
            if len(json.dumps(value, ensure_ascii=False)) > 500:
                result += "..."
        else:
            result += str(value)
        
        return result
    
    def _rewrite_query(self, user_query: str) -> str:
        """
        使用DeepSeek LLM改写查询，生成更好的检索关键词
        
        Args:
            user_query: 原始用户查询
        
        Returns:
            改写后的查询文本
        """
        system_prompt = """你是一个查询改写专家。你的任务是将用户的自然语言问题改写成更适合检索的关键词或查询语句。

改写原则：
1. 提取核心关键词和实体（人名、地点、事件等）
2. 将复杂问题拆解成多个子查询（用分号分隔）
3. 例如："A和B冲突的根本原因" -> "A的观点是什么;B的观点是什么;A和B冲突的事件是什么"
4. 保持原意，但更聚焦于可检索的关键信息
5. **重要**：对于需要推理的问题（如人物情感、关系发展、心理状态等），请明确要求基于证据的推理，在查询中强调需要检索"人物对话和情感描写的片段"来进行推断

只返回改写后的查询，不要添加解释。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"原始问题：{user_query}\n\n请改写为检索查询："}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            rewritten = response.choices[0].message.content.strip()
            logger.info(f"查询改写: {user_query} -> {rewritten}")
            return rewritten
            
        except Exception as e:
            logger.error(f"查询改写失败: {e}，使用原始查询")
            return user_query

    def _normalize_and_trim_history(self, conversation_history: Optional[List[Dict]], max_messages: int = 20, max_chars: int = 6000, current_query: Optional[str] = None) -> List[Dict]:
        """
        智能上下文管理：规范化并裁剪对话历史，实现对话记忆摘要、动态上下文选择、查询相关历史筛选
        
        - 只保留 role/content 字段
        - role 只允许 user/assistant（system 由服务端统一注入）
        - 短时记忆：最近3轮对话
        - 长时记忆：摘要后的关键信息（每5轮对话生成一次摘要）
        - 查询相关历史筛选：基于当前查询筛选相关历史轮次
        
        Args:
            conversation_history: 原始对话历史
            max_messages: 最大消息条数
            max_chars: 最大字符数
            current_query: 当前查询（用于相关历史筛选）
        """
        if not conversation_history:
            return []

        # 步骤1：规范化对话历史
        normalized: List[Dict] = []
        for item in conversation_history:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role not in ("user", "assistant"):
                continue
            if not isinstance(content, str):
                continue
            content = content.strip()
            if not content:
                continue
            normalized.append({"role": role, "content": content})

        if not normalized:
            return []

        # 步骤2：查询相关历史筛选（如果提供了当前查询）
        if current_query and len(normalized) > 6:
            # 提取当前查询中的关键实体（简单实现：提取人物名称和关键词）
            query_lower = current_query.lower()
            relevant_indices = []
            
            # 检查人物名称（如果人物画像库已加载）
            if hasattr(self, 'rag_system') and self.rag_system.character_names:
                for i, msg in enumerate(normalized):
                    msg_lower = msg["content"].lower()
                    # 检查是否包含相同的人物名称
                    for char_name in self.rag_system.character_names:
                        if char_name.lower() in query_lower and char_name.lower() in msg_lower:
                            relevant_indices.append(i)
                            break
            
            # 检查关键词匹配（提取查询中的关键词）
            query_keywords = [w for w in current_query.split() if len(w) > 1]
            for i, msg in enumerate(normalized):
                if i in relevant_indices:
                    continue
                msg_lower = msg["content"].lower()
                # 如果消息包含查询中的关键词，标记为相关
                matched_keywords = sum(1 for kw in query_keywords if kw.lower() in msg_lower)
                if matched_keywords >= len(query_keywords) * 0.3:  # 至少匹配30%的关键词
                    relevant_indices.append(i)
            
            # 如果找到相关历史，优先保留相关轮次
            if relevant_indices:
                relevant_indices = sorted(set(relevant_indices))
                # 保留相关轮次及其上下文（前后各1轮）
                expanded_indices = set()
                for idx in relevant_indices:
                    expanded_indices.add(idx)
                    if idx > 0:
                        expanded_indices.add(idx - 1)
                    if idx < len(normalized) - 1:
                        expanded_indices.add(idx + 1)
                
                # 构建筛选后的历史：相关轮次 + 最近3轮（短时记忆）
                short_term_count = 3
                recent_indices = set(range(max(0, len(normalized) - short_term_count * 2), len(normalized)))
                final_indices = sorted(expanded_indices | recent_indices)
                
                # 按原始顺序重新组织
                filtered_history = [normalized[i] for i in final_indices if i < len(normalized)]
                if filtered_history:
                    normalized = filtered_history
                    logger.info(f"[上下文筛选] 基于查询筛选出 {len(normalized)} 条相关历史")

        # 步骤3：短时记忆 - 保留最近3轮对话
        short_term_count = 3  # 3轮对话 = 6条消息（每轮包含user和assistant）
        short_term_history = normalized[-short_term_count * 2:] if len(normalized) > short_term_count * 2 else normalized
        
        # 步骤4：长时记忆 - 对话记忆摘要（每5轮对话生成一次摘要）
        long_term_summary = None
        if len(normalized) > 10:  # 如果历史超过10条消息（约5轮对话）
            # 提取早期对话（除了最近3轮）
            early_history = normalized[:-short_term_count * 2] if len(normalized) > short_term_count * 2 else []
            
            if early_history:
                # 生成摘要（简化实现：提取关键信息）
                try:
                    summary_prompt = f"""请从以下对话历史中提取关键信息，生成一个简洁的摘要。

对话历史：
{chr(10).join([f"{msg['role']}: {msg['content'][:200]}" for msg in early_history[-10:]])}

要求：
1. 提取用户偏好、已确认的信息、待澄清的问题
2. 保留重要的人物、事件、概念
3. 摘要长度控制在200字以内
4. 只返回摘要内容，不要添加额外说明

摘要："""
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "你是一个专业的对话摘要专家，擅长从对话历史中提取关键信息。"},
                            {"role": "user", "content": summary_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=300
                    )
                    
                    long_term_summary = response.choices[0].message.content.strip()
                    logger.info(f"[对话摘要] 生成长时记忆摘要，长度: {len(long_term_summary)} 字符")
                except Exception as e:
                    logger.warning(f"生成对话摘要失败: {e}")

        # 步骤5：合并短时记忆和长时记忆
        final_history = []
        
        # 如果有长时记忆摘要，添加为系统消息格式的摘要
        if long_term_summary:
            final_history.append({
                "role": "system",
                "content": f"[对话历史摘要] {long_term_summary}"
            })
        
        # 添加短时记忆（最近3轮对话）
        final_history.extend(short_term_history)
        
        # 步骤6：按字符数裁剪（如果仍然过长）
        if len(final_history) > max_messages:
            final_history = final_history[-max_messages:]
        
        # 再按字符裁剪（从尾部往前累计）
        total = 0
        trimmed_rev: List[Dict] = []
        for msg in reversed(final_history):
            c = msg["content"]
            if total + len(c) > max_chars and trimmed_rev:
                break
            trimmed_rev.append(msg)
            total += len(c)
            if total >= max_chars:
                break

        result = list(reversed(trimmed_rev))
        logger.info(f"[上下文管理] 最终保留 {len(result)} 条消息，总字符数: {total}")
        return result
    
    def _route_question(self, user_query: str) -> Tuple[bool, str, str, str]:
        """
        统一路由判断：在一次API调用中同时完成"是否需要RAG"和"问题类型"的决策
        
        Returns:
            (need_rag, question_type, type_reason, rag_reason)
            - need_rag: 是否需要RAG
            - question_type: 问题类型
            - type_reason: 类型判断理由
            - rag_reason: RAG判断理由
        """
        # 先使用关键词快速判断（作为备选）
        quick_type = self._question_classifier(user_query)
        
        # 构建统一的判断提示词
        system_prompt = """你是一个智能路由系统，需要同时判断：
1. 用户问题是否需要从小说文档中检索信息（need_rag）
2. 如果需要检索，应该使用哪种检索策略（question_type）

问题类型说明：
- "global_summary": 询问故事整体内容、主题、梗概、主要情节等宏观问题
- "timeline_db": 询问时间顺序、发展进程、时间脉络等问题
- "character_arc": 询问人物关系发展、感情线演变等问题
- "imagery": 询问意象、象征、隐喻、符号意义等概念性问题
- "meta_analysis": 询问创作意图、背景音乐、伏笔、作者解读等元文本问题
- "behavior_explanation": 询问人物行为解释、心理动机、行为模式等行为分析类问题
- "scenario_simulation": 询问假设性情节推演、如果...会怎样等情节推演类问题
- "chapter_rag": 询问具体人物、事件、细节等微观问题

判断标准：
1. 如果问题涉及小说中的人物、情节、事件、细节等具体内容，need_rag=true
2. 如果问题是一般性对话、问候、闲聊等，need_rag=false
3. 如果need_rag=false，question_type设为"none"
4. **重要**：对于需要推理的问题（如人物情感、心理状态、关系推断等），在生成查询query时，必须明确要求基于证据的推理，强调需要检索"人物对话和情感描写的片段"来进行推断
5. **行为分析类问题**：如果问题涉及"为什么这样做"、"行为动机"、"心理原因"、"是什么样的人"、"性格特点"、"心理特征"等，应归类为"behavior_explanation"
6. **情节推演类问题**：如果问题涉及"如果...会怎样"、"假设"、"要是"等，应归类为"scenario_simulation"
7. **元文本分析类问题**：如果问题涉及"作者"、"创作意图"、"背景音乐"、"伏笔"、"作者解读"、"作者是个什么样的人"等，应归类为"meta_analysis"（元文本知识库包含作者的创作说明和解读）
8. **特别说明**：关于"作者"的问题（如"作者是个什么样的人"、"作者的性格"、"作者的创作风格"等）应该归类为"meta_analysis"或"behavior_explanation"，因为元文本知识库包含作者的创作说明和解读，行为分析模块可以分析作者的心理特征

只返回JSON格式：{"need_rag": true/false, "question_type": "类型", "type_reason": "类型判断理由", "rag_reason": "RAG判断理由"}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"用户问题：{user_query}\n\n请同时判断是否需要RAG和问题类型。"}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            need_rag = result.get("need_rag", True)  # 默认需要RAG（更安全）
            question_type = result.get("question_type", quick_type)  # 如果解析失败，使用关键词判断
            type_reason = result.get("type_reason", "LLM判断")
            rag_reason = result.get("rag_reason", "需要检索小说内容")
            
            # 验证问题类型是否有效
            valid_types = ["global_summary", "timeline_db", "character_arc", "imagery", "meta_analysis", "behavior_explanation", "scenario_simulation", "chapter_rag", "none"]
            if question_type not in valid_types:
                question_type = quick_type
                type_reason = "类型无效，使用关键词匹配"
            
            # 如果不需要RAG，问题类型设为none
            if not need_rag:
                question_type = "none"
            
            # 不在_route_question内部输出详细日志，由调用方决定如何输出
            # 这样可以避免重复日志，调用方可以添加模块标识
            return need_rag, question_type, type_reason, rag_reason
            
        except Exception as e:
            logger.error(f"统一路由判断失败: {e}，使用关键词判断")
            # 如果判断失败，使用关键词判断结果
            need_rag = True  # 默认需要RAG（更安全）
            question_type = quick_type
            return need_rag, question_type, "关键词匹配", "判断失败，默认使用RAG"
    
    def _multi_hop_retrieve(self, query_chain: List[Dict], question_type: str, user_query: str, progress_callback=None) -> Dict:
        """
        多跳检索：按查询链顺序执行检索，并将结果合并
        
        Args:
            query_chain: 查询链，格式: [{"step": 1, "query": "...", "intent": "...", "depends_on": null}, ...]
            question_type: 问题类型
            user_query: 原始用户查询
            progress_callback: 进度回调函数，接收 (current_step, total_steps) 参数
        
        Returns:
            合并后的检索结果字典
        """
        logger.info("=" * 60)
        logger.info("开始多跳检索...")
        logger.info(f"查询链长度: {len(query_chain)}")
        logger.info("=" * 60)
        
        all_contexts = []
        all_sources = []
        previous_results = {}  # 存储前一步的检索结果，供后续步骤使用
        
        for i, query_item in enumerate(query_chain, 1):
            step = query_item.get("step", i)
            query = query_item.get("query", "")
            intent = query_item.get("intent", "")
            depends_on = query_item.get("depends_on")
            
            logger.info(f"\n[步骤 {step}/{len(query_chain)}] {intent}")
            logger.info(f"查询: {query}")
            if depends_on:
                logger.info(f"依赖步骤: {depends_on}")
            
            # 发送进度更新
            if progress_callback:
                try:
                    progress_callback(step, len(query_chain))
                except Exception as e:
                    logger.warning(f"进度回调失败: {e}")
            
            # 如果依赖前一步的结果，将前一步的上下文添加到当前查询中
            enhanced_query = query
            if depends_on and depends_on in previous_results:
                prev_context = previous_results[depends_on].get('context', '')
                if prev_context:
                    # 将前一步的关键信息添加到查询中，帮助当前检索
                    enhanced_query = f"{query}\n\n[基于前一步检索到的信息：{prev_context[:300]}...]"
                    logger.info(f"已增强查询（包含前一步结果）")
            
            # 执行检索
            try:
                # 对每个查询进行改写
                rewritten_query = self._rewrite_query(enhanced_query)
                
                # 执行检索
                step_result = self._retrieve_context(rewritten_query, question_type, original_query=user_query)
                
                step_context = step_result.get("context", "")
                step_sources = step_result.get("sources", [])
                
                # 标记来源
                if step_context:
                    marked_context = f"[步骤{step}检索结果 - {intent}]\n{step_context}"
                    all_contexts.append(marked_context)
                    all_sources.extend(step_sources)
                    
                    # 保存当前步骤的结果，供后续步骤使用
                    previous_results[step] = {
                        'context': step_context,
                        'sources': step_sources
                    }
                    
                    logger.info(f"步骤{step}检索完成，获得 {len(step_context)} 字符的上下文")
                else:
                    logger.warning(f"步骤{step}检索未获得上下文")
                    
            except Exception as e:
                logger.error(f"步骤{step}检索失败: {e}")
                continue
        
        # 合并所有步骤的上下文，并去重相同段落
        # 使用简化的去重策略：比较整个上下文块的内容
        deduplicated_contexts = []
        seen_content = set()
        
        for context in all_contexts:
            # 提取实际内容（去除步骤标记，只比较内容部分）
            if context.startswith("[步骤"):
                # 提取步骤标记后的内容
                lines = context.split("\n", 1)
                step_marker = lines[0] if len(lines) > 0 else ""
                content = lines[1] if len(lines) > 1 else ""
            else:
                step_marker = ""
                content = context
            
            # 使用内容的hash作为唯一标识（去除空白字符和换行符）
            # 使用前1000字符作为key，避免过长
            content_key = content.strip().replace(" ", "").replace("\n", "").replace("\t", "")[:1000]
            
            # 如果内容为空或已存在，跳过
            if content_key and content_key not in seen_content:
                seen_content.add(content_key)
                # 如果有步骤标记，保留它
                if step_marker:
                    deduplicated_contexts.append(f"{step_marker}\n{content}")
                else:
                    deduplicated_contexts.append(context)
        
        combined_context = "\n\n".join(deduplicated_contexts) if deduplicated_contexts else ""
        
        # 记录去重信息
        original_count = len(all_contexts)
        deduplicated_count = len(deduplicated_contexts)
        if original_count > deduplicated_count:
            logger.info(f"去重：从 {original_count} 个上下文块中移除了 {original_count - deduplicated_count} 个重复块")
        
        # 去重来源（基于内容）
        seen_sources = set()
        unique_sources = []
        for source in all_sources:
            source_key = source.get('content', '')[:100]  # 使用内容前100字符作为唯一标识
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_sources.append(source)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"多跳检索完成，共 {len(query_chain)} 步")
        logger.info(f"合并上下文长度: {len(combined_context)} 字符（去重后）")
        logger.info(f"合并来源数量: {len(unique_sources)}")
        logger.info("=" * 60 + "\n")
        
        return {
            "context": combined_context,
            "sources": unique_sources,
            "target_source": "雪落成诗",  # 默认值，实际应该从检索结果中获取
            "used_dynamic_summary": False,
            "multi_hop": True,
            "query_chain": query_chain
        }
    
    def _multi_hop_retrieve_with_progress(self, query_chain: List[Dict], question_type: str, user_query: str):
        """
        多跳检索（生成器版本）：按查询链顺序执行检索，并在每个步骤发送进度更新
        
        Args:
            query_chain: 查询链，格式: [{"step": 1, "query": "...", "intent": "...", "depends_on": null}, ...]
            question_type: 问题类型
            user_query: 原始用户查询
        
        Yields:
            进度更新字典: {"type": "multi_hop_progress", "message": "..."}
            最终结果字典: {"type": "multi_hop_result", "result": {...}}
        """
        logger.info("=" * 60)
        logger.info("开始多跳检索（带进度更新）...")
        logger.info(f"查询链长度: {len(query_chain)}")
        logger.info("=" * 60)
        
        all_contexts = []
        all_sources = []
        previous_results = {}  # 存储前一步的检索结果，供后续步骤使用
        
        for i, query_item in enumerate(query_chain, 1):
            step = query_item.get("step", i)
            query = query_item.get("query", "")
            intent = query_item.get("intent", "")
            depends_on = query_item.get("depends_on")
            
            logger.info(f"\n[步骤 {step}/{len(query_chain)}] {intent}")
            logger.info(f"查询: {query}")
            if depends_on:
                logger.info(f"依赖步骤: {depends_on}")
            
            # 发送进度更新
            yield {
                "type": "multi_hop_progress",
                "message": f"您的问题需要深度思考，正在执行深度思考推理（{step}/{len(query_chain)}）…"
            }
            
            # 如果依赖前一步的结果，将前一步的上下文添加到当前查询中
            enhanced_query = query
            if depends_on and depends_on in previous_results:
                prev_context = previous_results[depends_on].get('context', '')
                if prev_context:
                    # 将前一步的关键信息添加到查询中，帮助当前检索
                    enhanced_query = f"{query}\n\n[基于前一步检索到的信息：{prev_context[:300]}...]"
                    logger.info(f"已增强查询（包含前一步结果）")
            
            # 执行检索
            try:
                # 对每个查询进行改写
                rewritten_query = self._rewrite_query(enhanced_query)
                
                # 执行检索
                step_result = self._retrieve_context(rewritten_query, question_type, original_query=user_query)
                
                step_context = step_result.get("context", "")
                step_sources = step_result.get("sources", [])
                
                # 标记来源
                if step_context:
                    marked_context = f"[步骤{step}检索结果 - {intent}]\n{step_context}"
                    all_contexts.append(marked_context)
                    all_sources.extend(step_sources)
                    
                    # 保存当前步骤的结果，供后续步骤使用
                    previous_results[step] = {
                        'context': step_context,
                        'sources': step_sources
                    }
                    
                    logger.info(f"步骤{step}检索完成，获得 {len(step_context)} 字符的上下文")
                else:
                    logger.warning(f"步骤{step}检索未获得上下文")
                    
            except Exception as e:
                logger.error(f"步骤{step}检索失败: {e}")
                continue
        
        # 合并所有步骤的上下文，并去重相同段落
        # 使用简化的去重策略：比较整个上下文块的内容
        deduplicated_contexts = []
        seen_content = set()
        
        for context in all_contexts:
            # 提取实际内容（去除步骤标记，只比较内容部分）
            if context.startswith("[步骤"):
                # 提取步骤标记后的内容
                lines = context.split("\n", 1)
                step_marker = lines[0] if len(lines) > 0 else ""
                content = lines[1] if len(lines) > 1 else ""
            else:
                step_marker = ""
                content = context
            
            # 使用内容的hash作为唯一标识（去除空白字符和换行符）
            # 使用前1000字符作为key，避免过长
            content_key = content.strip().replace(" ", "").replace("\n", "").replace("\t", "")[:1000]
            
            # 如果内容为空或已存在，跳过
            if content_key and content_key not in seen_content:
                seen_content.add(content_key)
                # 如果有步骤标记，保留它
                if step_marker:
                    deduplicated_contexts.append(f"{step_marker}\n{content}")
                else:
                    deduplicated_contexts.append(context)
        
        combined_context = "\n\n".join(deduplicated_contexts) if deduplicated_contexts else ""
        
        # 记录去重信息
        original_count = len(all_contexts)
        deduplicated_count = len(deduplicated_contexts)
        if original_count > deduplicated_count:
            logger.info(f"去重：从 {original_count} 个上下文块中移除了 {original_count - deduplicated_count} 个重复块")
        
        # 去重来源（基于内容）
        seen_sources = set()
        unique_sources = []
        for source in all_sources:
            source_key = source.get('content', '')[:100]  # 使用内容前100字符作为唯一标识
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_sources.append(source)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"多跳检索完成，共 {len(query_chain)} 步")
        logger.info(f"合并上下文长度: {len(combined_context)} 字符（去重后）")
        logger.info(f"合并来源数量: {len(unique_sources)}")
        logger.info("=" * 60 + "\n")
        
        # 返回最终结果
        result = {
            "context": combined_context,
            "sources": unique_sources,
            "target_source": "雪落成诗",  # 默认值，实际应该从检索结果中获取
            "used_dynamic_summary": False,
            "multi_hop": True,
            "query_chain": query_chain
        }
        
        yield {
            "type": "multi_hop_result",
            "result": result
        }
    
    def _retrieve_context(self, query: str, question_type: str = "chapter_rag", original_query: Optional[str] = None) -> Dict:
        """
        根据问题类型检索相关上下文（调用各个RAG模块）
        
        Args:
            query: 查询文本（已改写，用于 RAG/时间线等检索）
            question_type: 问题类型
            original_query: 用户原始问题（用于人物检测，避免改写后「男女主角」等词丢失）
        
        Returns:
            检索结果字典，包含时间统计信息
        """
        import time
        rag_start_time = time.time()
        logger.info("=" * 60)
        logger.info("开始RAG检索...")
        logger.info(f"查询: {query}")
        logger.info(f"问题类型: {question_type}")
        logger.info("=" * 60)
        
        # ========== 判断是否需要动态摘要 ==========
        if self.dynamic_summary_generator:
            use_dynamic_summary, dynamic_reason = self.dynamic_summary_generator.should_use(query, question_type)
            if use_dynamic_summary:
                logger.info(f"触发动态摘要流程: {dynamic_reason}")
                result = self.dynamic_summary_generator.retrieve(query)
                if not result.get('fallback', True):
                    logger.info("动态摘要生成成功，使用动态摘要作为上下文")
                    return {
                        'context': result['context'],
                        'sources': result['sources'],
                        'target_source': result.get('target_source', '雪落成诗'),
                        'used_dynamic_summary': True
                    }
                else:
                    logger.warning("动态摘要质量不足，回退到常规检索流程")
        
        # ========== 优化：涉及人物的问题先检查人物画像库 ==========
        # 步骤1：检查问题是否涉及人物（用原始问句检测，避免改写后丢失「男女主角」「谁」等）
        character_results = None
        character_context = ""
        character_sources = []
        is_character_related = False
        character_query = (original_query or query).strip()
        character_keywords = ["主角", "男女主角", "人物", "谁", "姓名", "角色", "男主", "女主"]
        is_likely_character_question = any(kw in character_query for kw in character_keywords)
        
        if self.rag_system.character_profiles:
            character_results = self.rag_system.search_characters(character_query, top_k=5)
            # 人物类问题（含「主角」「谁」等）用较低阈值，否则用 5.0
            threshold = 2.0 if is_likely_character_question else 5.0
            if character_results and character_results[0].get('score', 0) > threshold:
                is_character_related = True
                logger.info(f"[人物检测] 检测到问题涉及人物，找到 {len(character_results)} 个相关人物（score>{threshold}）")
                # 构建人物画像上下文
                character_context_parts = []
                for char_result in character_results:
                    character_context_parts.append(char_result['content'])
                    character_sources.append({
                        "source": "人物画像",
                        "character": char_result['character'],
                        "content": char_result['content'][:200],
                        "hide_in_ui": True  # 标记为隐藏，不在UI中显示
                    })
                character_context = "\n\n".join(character_context_parts)
                logger.info(f"[人物画像] 已提取人物画像信息，包含 {len(character_sources)} 个人物")
        
        # ========== 并行检索：RAG检索 + 人物画像检索（如果已检测到人物） ==========
        def retrieve_rag():
            """RAG检索任务（优先走 LangChain 编排，便于可观测与换存储/模型）"""
            # 行为分析类问题使用行为分析模块（不走 LangChain 适配，保持原有逻辑）
            if question_type in ["behavior_explanation", "scenario_simulation"]:
                retriever = self.retrievers.get(question_type)
                result = retriever.retrieve(query, question_type=question_type)
            elif self._lc_retriever_map and documents_to_result and question_type in self._lc_retriever_map:
                # LangChain 编排：通过 BaseRetriever 统一调用
                lc_ret = self._lc_retriever_map.get(question_type, self._lc_retriever_map["chapter_rag"])
                try:
                    docs = lc_ret.invoke(query)
                    result = documents_to_result(docs)
                except Exception as e:
                    logger.warning(f"LangChain 检索调用异常: {e}，回退到直接调用")
                    retriever = self.retrievers.get(question_type, self.retrievers["chapter_rag"])
                    result = retriever.retrieve(query)
            else:
                retriever = self.retrievers.get(question_type, self.retrievers["chapter_rag"])
                result = retriever.retrieve(query)
            
            # 如果检索失败，回退到章节RAG（行为分析类问题不回退）
            if result.get('fallback', False) and question_type not in ["chapter_rag", "behavior_explanation", "scenario_simulation"]:
                logger.warning(f"{question_type}检索失败，回退到章节RAG...")
                result = self.retrievers["chapter_rag"].retrieve(query)
            
            # 处理章节摘要回退
            if question_type == "global_summary" and result.get('fallback', False):
                logger.warning("全局摘要为空，回退到章节摘要...")
                result = self.retrievers["chapter_summary"].retrieve(query)
                if result.get('fallback', False):
                    logger.warning("章节摘要也失败，回退到章节RAG...")
                    result = self.retrievers["chapter_rag"].retrieve(query)
            
            return result
        
        def retrieve_characters_parallel():
            """人物画像检索任务（并行执行，如果之前未检测到）"""
            if is_character_related:
                # 如果已经检测到人物，直接返回之前的结果
                return {
                    'context': character_context,
                    'sources': character_sources,
                    'score': character_results[0]['score'] if character_results else 0.0
                }
            
            try:
                char_results = self.rag_system.search_characters(character_query, top_k=5)
                if char_results:
                    # 构建人物画像上下文
                    char_context_parts = []
                    char_sources = []
                    for char_result in char_results:
                        char_context_parts.append(char_result['content'])
                        char_sources.append({
                            "source": "人物画像",
                            "character": char_result['character'],
                            "content": char_result['content'][:200],
                            "hide_in_ui": True  # 标记为隐藏，不在UI中显示
                        })
                    
                    return {
                        'context': "\n\n".join(char_context_parts),
                        'sources': char_sources,
                        'score': char_results[0]['score'] if char_results else 0.0
                    }
                else:
                    return {
                        'context': '',
                        'sources': [],
                        'score': 0.0
                    }
            except Exception as e:
                logger.warning(f"人物画像检索失败: {e}")
                return {
                    'context': '',
                    'sources': [],
                    'score': 0.0
                }
        
        # 对于行为分析类问题，需要同时获取行为分析模块的上下文和RAG检索的原文片段
        if question_type in ["behavior_explanation", "scenario_simulation"]:
            # 先进行RAG检索获取原文片段（用于事实核查）
            rag_retriever = self.retrievers.get("chapter_rag")
            rag_result = rag_retriever.retrieve(query)
            rag_original_text = rag_result.get('context', '')
            rag_sources = rag_result.get('sources', [])
            
            # 获取行为分析模块的上下文（传入RAG原文片段，让其按新顺序组织）
            behavior_retriever = self.retrievers.get(question_type)
            behavior_result = behavior_retriever.retrieve(query, question_type=question_type, rag_original_text=rag_original_text)
            behavior_context = behavior_result.get('context', '')
            behavior_sources = behavior_result.get('sources', [])
            additional_knowledge = behavior_result.get('additional_knowledge')
            
            # 合并上下文和来源
            combined_context = behavior_context
            combined_sources = behavior_sources + rag_sources
        else:
            # 步骤2和3：同时进行RAG检索（如果已检测到人物，人物画像已准备好）
            # 并行执行RAG检索和人物画像检索（如果之前未检测到）
            if is_character_related:
                # 如果已经检测到人物，只执行RAG检索
                logger.info("[优化流程] 已检测到人物，并行执行RAG检索...")
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future_rag = executor.submit(retrieve_rag)
                    try:
                        result = future_rag.result()
                    except Exception as e:
                        if "Nothing found on disk" in str(e) or "rebuild_index" in str(e) or "ChromaDB" in str(e):
                            logger.error(f"向量索引损坏: {e}")
                            result = {
                                'context': '【系统提示】ChromaDB 向量索引损坏或缺失，无法检索。请关闭应用后运行 rebuild_index.py 重建索引。',
                                'sources': [],
                                'fallback': True
                            }
                        else:
                            raise
                
                # 使用之前提取的人物画像信息
                character_result = {
                    'context': character_context,
                    'sources': character_sources,
                    'score': character_results[0]['score'] if character_results else 0.0
                }
            else:
                # 并行执行RAG检索和人物画像检索
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_rag = executor.submit(retrieve_rag)
                    future_characters = executor.submit(retrieve_characters_parallel)
                    
                    # 等待结果（捕获 ChromaDB 索引损坏错误，返回友好提示）
                    try:
                        result = future_rag.result()
                    except Exception as e:
                        if "Nothing found on disk" in str(e) or "rebuild_index" in str(e) or "ChromaDB" in str(e):
                            logger.error(f"向量索引损坏: {e}")
                            result = {
                                'context': '【系统提示】ChromaDB 向量索引损坏或缺失，无法检索。请关闭应用后运行 rebuild_index.py 重建索引。',
                                'sources': [],
                                'fallback': True
                            }
                        else:
                            raise
                    character_result = future_characters.result()
            
            # 步骤3：将人物画像库中的结构化信息和RAG检索的文档片段一起提供给LLM
            rag_context = result.get('context', '')
            char_context = character_result.get('context', '')
            
            # 步骤3.1：事实类问题（chapter_rag）并入时间线索引与关键事件链条，用于事实核查
            timeline_context = ""
            timeline_sources = []
            event_chain_context = ""
            event_chain_sources = []
            timeline_result = None
            event_chain_results = []
            if question_type == "chapter_rag":
                try:
                    timeline_retriever = self.retrievers.get("timeline_db")
                    timeline_result = timeline_retriever.retrieve(query, top_k=5)
                    if timeline_result and not timeline_result.get('fallback', True) and timeline_result.get('context'):
                        timeline_context = timeline_result.get('context', '')
                        timeline_sources = timeline_result.get('sources', [])
                        logger.info("[事实核查] 已并入时间线索引，用于事实类问题的核查")
                except Exception as e:
                    logger.warning(f"[事实核查] 时间线检索异常，跳过: {e}")
                try:
                    event_chain_results = self.rag_system.search_event_chain(query, top_k=4)
                    if event_chain_results:
                        parts = []
                        for r in event_chain_results:
                            line = r.get("content")
                            if not line:
                                line = f"- [{r.get('title', '')}]（{r.get('source', '')}）{r.get('behavior_description', '')[:200]}"
                                if r.get('consequences'):
                                    line += "；后果: " + "、".join(r['consequences'][:2])
                            else:
                                line = "- " + line
                            parts.append(line)
                        event_chain_context = "\n".join(parts)
                        event_chain_sources = [{"source": "关键事件链条", "content": (r.get('title') or '') + ' ' + (r.get('source') or ''), "hide_in_ui": True} for r in event_chain_results]
                        logger.info("[事实核查] 已并入关键事件链条，用于事实类问题的核查")
                except Exception as e:
                    logger.warning(f"[事实核查] 关键事件链条检索异常，跳过: {e}")
            
            # 综合时间线/人物画像/关键事件链条的章节来源，优先取对应原文块（不硬编码具体问题）
            preferred_chapters = []
            if isinstance(timeline_result, dict):
                preferred_chapters.extend(timeline_result.get('preferred_chapters') or [])
            import re
            for m in re.finditer(r'来源[：:]\s*([^\n]+)', char_context or ""):
                ch = m.group(1).strip()
                if ch and ch not in preferred_chapters:
                    preferred_chapters.append(ch)
            for r in (event_chain_results or []):
                src = r.get("source") or ""
                if src:
                    fmt = self.rag_system._format_chapter_source(src)
                    if fmt and fmt not in preferred_chapters:
                        preferred_chapters.append(fmt)
            preferred_chapters = list(dict.fromkeys(preferred_chapters))
            if preferred_chapters:
                try:
                    chapter_chunks = self.rag_system.get_chunks_by_chapter(preferred_chapters, max_per_chapter=5)
                    if chapter_chunks:
                        chapter_block = "\n\n".join([f"[来源: {c.get('metadata', {}).get('chapter', '')}]\n{c['content']}" for c in chapter_chunks])
                        rag_context = f"[与时间线/人物/事件链对应的原文]\n{chapter_block}\n\n[文档检索 - 原文片段]\n{rag_context}" if rag_context else f"[与时间线/人物/事件链对应的原文]\n{chapter_block}"
                        logger.info(f"[章节对齐] 已按时间线/人物/事件链来源取块，章节数: {len(preferred_chapters)}，块数: {len(chapter_chunks)}")
                except Exception as e:
                    logger.warning(f"[章节对齐] 按章节取块异常，跳过: {e}")
            
            # 合并上下文（时间线 > 关键事件链条 > 人物画像 > 文档检索）
            if char_context:
                # 如果RAG检索有结果，将人物画像放在前面作为结构化信息
                if rag_context:
                    combined_context = f"[人物画像库 - 结构化信息]\n{char_context}\n\n[文档检索 - 原文片段]\n{rag_context}"
                    logger.info("[信息融合] 已将人物画像库的结构化信息和RAG检索的文档片段合并")
                else:
                    combined_context = f"[人物画像库 - 结构化信息]\n{char_context}"
                    logger.info("[信息融合] 仅使用人物画像库的结构化信息")
                
                # 合并来源（人物画像来源优先）
                combined_sources = character_result.get('sources', []) + result.get('sources', [])
            else:
                combined_context = rag_context
                combined_sources = result.get('sources', [])
            
            # 事实类问题：将时间线索引与关键事件链条置于最前（顺序：时间线 > 关键事件链条 > 人物/文档）
            if event_chain_context:
                combined_context = f"[关键事件链条 - 用于事实核查]\n{event_chain_context}\n\n{combined_context}"
                combined_sources = event_chain_sources + combined_sources
            if timeline_context:
                combined_context = f"[时间线索引 - 用于事实核查]\n{timeline_context}\n\n{combined_context}"
                combined_sources = timeline_sources + combined_sources
        
        rag_end_time = time.time()
        rag_elapsed_time = rag_end_time - rag_start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("RAG检索完成")
        if question_type not in ["behavior_explanation", "scenario_simulation"]:
            if 'character_result' in locals() and character_result.get('context'):
                logger.info(f"人物画像检索: 找到 {len(character_result.get('sources', []))} 个人物")
        logger.info(f"RAG检索耗时: {rag_elapsed_time:.2f} 秒")
        logger.info("=" * 60 + "\n")
        
        # 处理additional_knowledge
        final_additional_knowledge = None
        if question_type in ["behavior_explanation", "scenario_simulation"]:
            final_additional_knowledge = additional_knowledge if 'additional_knowledge' in locals() else None
        else:
            final_additional_knowledge = result.get('additional_knowledge') if 'result' in locals() else None
        
        return {
            "context": combined_context,
            "sources": combined_sources,
            "target_source": result.get('target_source', '雪落成诗') if 'result' in locals() else (behavior_result.get('target_source', '雪落成诗') if question_type in ["behavior_explanation", "scenario_simulation"] and 'behavior_result' in locals() else '雪落成诗'),
            "used_dynamic_summary": False,
            "additional_knowledge": final_additional_knowledge,
            "rag_time": rag_elapsed_time  # 添加RAG检索耗时
        }
    
    def chat(self, user_query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        处理用户查询，智能判断是否需要RAG
        
        Args:
            user_query: 用户查询
            conversation_history: 对话历史，格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
        Returns:
            包含回答、来源等信息的字典
        """
        conversation_history = self._normalize_and_trim_history(conversation_history, current_query=user_query)
        
        # 统一路由判断：同时完成"是否需要RAG"和"问题类型"的决策
        need_rag, question_type, type_reason, rag_reason = self._route_question(user_query)
        logger.info(f"路由判断: need_rag={need_rag}, question_type={question_type}")
        logger.info(f"  类型判断理由: {type_reason}")
        logger.info(f"  RAG判断理由: {rag_reason}")
        
        context = ""
        sources = []
        target_source = None
        
        if need_rag and question_type != "none":
            # ========== 只对行为解释类和情节推演类问题使用Reasoning模块判断是否需要多跳检索 ==========
            # 判断是否为行为分析类问题（需要多跳推理）
            is_behavior_analysis_type = question_type in ["behavior_explanation", "scenario_simulation"]
            
            if is_behavior_analysis_type and self.reasoning_module:
                # 只对行为解释类和情节推演类问题使用reasoning模块
                try:
                    # 先进行一次初始检索，获取初始上下文用于reasoning判断
                    rewrite_start_time = time.time()
                    rewritten_query = self._rewrite_query(user_query)
                    rewrite_time = time.time() - rewrite_start_time
                    initial_result = self._retrieve_context(rewritten_query, question_type, original_query=user_query)
                    initial_context = initial_result.get("context", "")
                    
                    # 调用reasoning模块判断当前推理步骤数量是否充足
                    reasoning_result = self.reasoning_module.reason(
                        user_query=user_query,
                        question_type=question_type,
                        initial_context=initial_context,
                        conversation_history=conversation_history
                    )
                    
                    need_multi_hop = reasoning_result.get("need_secondary_retrieval", False)
                    query_chain = reasoning_result.get("query_chain", [])
                    
                    logger.info(f"[行为分析类] Reasoning判断结果: need_multi_hop={need_multi_hop}")
                    logger.info(f"  推理说明: {reasoning_result.get('reasoning', '')}")
                    
                    if need_multi_hop and query_chain and len(query_chain) > 1:
                        # 使用多跳检索
                        logger.info("使用多跳检索模式")
                        rag_result = self._multi_hop_retrieve(query_chain, question_type, user_query)
                        context = rag_result["context"]
                        sources = rag_result["sources"]
                        target_source = rag_result.get("target_source", "雪落成诗")
                        # 保存时间统计
                        if isinstance(rag_result, dict):
                            rag_result["rewrite_time"] = rewrite_time
                            rag_result["rag_time"] = initial_result.get("rag_time", 0.0) + rag_result.get("rag_time", 0.0)
                    else:
                        # 使用单次检索（使用初始检索结果）
                        logger.info("使用单次检索模式（推理步骤充足）")
                        context = initial_result["context"]
                        sources = initial_result["sources"]
                        target_source = initial_result.get("target_source", "雪落成诗")
                        # 保存时间统计
                        rag_result = initial_result.copy() if isinstance(initial_result, dict) else {}
                        rag_result["rewrite_time"] = rewrite_time
                        if "rag_time" not in rag_result:
                            rag_result["rag_time"] = initial_result.get("rag_time", 0.0) if isinstance(initial_result, dict) else 0.0
                        
                except Exception as e:
                    logger.error(f"Reasoning模块调用失败: {e}，回退到单次检索")
                    # 回退到单次检索
                    rewrite_start_time = time.time()
                    rewritten_query = self._rewrite_query(user_query)
                    rewrite_time = time.time() - rewrite_start_time
                    rag_result = self._retrieve_context(rewritten_query, question_type, original_query=user_query)
                    context = rag_result["context"]
                    sources = rag_result["sources"]
                    target_source = rag_result["target_source"]
                    # 保存时间统计
                    if isinstance(rag_result, dict):
                        rag_result["rewrite_time"] = rewrite_time
            else:
                # 其他问题类型直接使用单次检索（不需要reasoning判断）
                logger.info(f"[{question_type}] 非行为分析类问题，直接使用单次检索")
                rewritten_query = self._rewrite_query(user_query)
                rag_result = self._retrieve_context(rewritten_query, question_type, original_query=user_query)
                context = rag_result["context"]
                sources = rag_result["sources"]
                target_source = rag_result["target_source"]
        
        # 判断是否为元文本分析问题
        is_meta_analysis = question_type == "meta_analysis" if need_rag and question_type else False
        
        # 判断是否为行为分析类问题
        is_behavior_analysis = question_type in ["behavior_explanation", "scenario_simulation"] if need_rag and question_type else False
        
        # 检查context中是否包含结构化数据源
        has_structured_sources = False
        if context:
            structured_markers = [
                "[作者独家解读]",
                "[人物画像库 - 结构化信息]",
                "[时间线:",
                "【1. 时间线索引】",
                "【3. 元文本分析库】",
                "【4. 相关人物心理模型】",
                "【6. 意象系统】"
            ]
            has_structured_sources = any(marker in context for marker in structured_markers)
        
        # 构建系统提示词（强调必须利用历史保持一致性）
        if is_meta_analysis:
            system_prompt = """你是一个得到作者真传的文学分析助手，专门回答关于《雪落成诗》和《影化成殇》这两部小说的创作意图、背景音乐、伏笔等元文本分析问题。

你的任务：
1. 严格依据作者提供的独家解读进行回答
2. 将作者解读作为核心依据，结合原文证据进行深入分析
3. 回答要深入、专业、有洞察力
4. 可以引用原文内容作为证据支撑

注意：
- 作者解读是权威上下文，必须优先使用
- 原文内容作为补充证据，用于支撑作者解读
- 回答要体现对创作技巧和意图的深度理解
- **信息来源规范（必须遵守）**：对用户呈现时，信息来源一律只写「原文」或「作品原文」。严禁在回答中出现时间线、人物画像、元文本知识库、关键事件链条、作者解读等任何内部数据源名称；只使用其内容作答，若需说明来源则仅写「原文」。
- **必须结合对话历史**：延续上下文、指代一致、不要把同一会话当作单轮问答"""
        else:
            system_prompt = f"""你是一个专业的小说问答助手，专门回答关于《雪落成诗》和《影化成殇》这两部小说的问题。

你的任务：
1. 基于提供的文档内容回答问题
2. 如果文档中没有相关信息，诚实地说不知道
3. 回答要准确、简洁、有依据
4. 可以适当引用原文内容，但要自然流畅

注意：
- 《雪落成诗》是主体小说
- 《影化成殇》是续集，分析时需要结合《雪落成诗》的背景，但只输出《影化成殇》中直接提及的内容
- **章节引用规范**：在引用章节时，请使用完整的章节名称，例如：
  * "第1卷 情感主线 第1章 缘起，云散"（而不是"1.1"或"第1卷第1章"）
  * "第2卷 人物篇 第1章 天台，隧道"（而不是"2.1"或"第2卷第1章"）
  * "第0卷 序章 雪落惊梦"（而不是"序章"或"0.1"）
- **信息来源规范（必须遵守）**：对用户呈现时，信息来源一律只写「原文」或「作品原文」。严禁在回答中出现时间线、人物画像、元文本知识库、关键事件链条等任何内部数据源名称；只使用其内容作答，若需说明来源则仅写「原文」。
- 保持回答的语言表述自然、流畅，符合中文阅读习惯
- **必须结合对话历史**：延续上下文、指代一致、不要把同一会话当作单轮问答"""

        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加对话历史
        if conversation_history:
            messages.extend(conversation_history)
        
        # 添加当前查询和上下文
        if context:
            if is_meta_analysis:
                # 元文本分析的提示词格式
                # 分离作者解读和原文补充
                if '[来源:' in context:
                    parts = context.split('[来源:')
                    author_interpretation = parts[0].replace('[作者独家解读]', '').strip()
                    original_text = '\n\n[来源:'.join(parts[1:]) if len(parts) > 1 else ''
                else:
                    author_interpretation = context.replace('[作者独家解读]', '').strip() if '[作者独家解读]' in context else ''
                    original_text = context
                
                user_message = f"""你是一个得到作者真传的专业文学分析助手，擅长从文本细节中提炼人物的深层心理模式和行为逻辑。请严格依据以下材料回答问题。

【作者独家解读】
{author_interpretation}

【作品原文补充】
{original_text}

问题：{user_query}

分析要求：
1. 请超越表面标签，挖掘人物行为背后的可迁移模式与底层认知逻辑。
2. 如果问题涉及假设推演，请基于人物的心理机制和已知行为模式，给出合理的可能性分析。
3. 分析需紧扣提供的证据，但结论应具有概括性，能够解释人物的其他类似行为。

**信息来源规范**：回答中若需说明信息来源，一律只写「原文」或「作品原文」，不得出现元文本知识库、作者独家解读等任何内部名称。"""
            elif is_behavior_analysis:
                # 行为分析类问题的提示词格式（统一格式，强调事实核查；动机优先采信人物自述）
                user_message = f"""基于以下上下文数据回答用户问题：

{context}

问题：{user_query}

**动机采信原则**：在分析行为动机时，优先采信人物自述的动机，其次再结合外部事件关联。

请严格按照以下流程回答：
1. **首先进行事实核查**：定位问题中核心事件的具体时间点，并从原文片段中引用1-2句直接证据。
2. **然后进行模式识别与分析**：在事实基础上进行分析，确保分析与事实一致；动机解释以人物自述为准。
3. **最后组织答案**：直接给出分析结果，不要体现推理步骤和具体来源。分步拆解机制，连接其他关联事件展现模式一致性。

**信息来源规范**：回答中若需说明信息来源，一律只写「原文」或「作品原文」，不得出现时间线、人物画像、元文本分析库、意象系统、心理学概念库、情节推演规则等任何内部名称。不要用要点形式体现流程。"""
            else:
                structured_source_hint = ""
                if has_structured_sources:
                    structured_source_hint = "\n\n**信息来源规范**：回答中若需说明信息来源，一律只写「原文」或「作品原文」，不得出现任何内部数据源名称。"
                fact_check_hint = ""
                if "[时间线索引" in (context or "") or "时间线:" in (context or "") or "[关键事件链条" in (context or ""):
                    fact_check_hint = (
                        "\n\n**事实核查**："
                        "若问具体事件时间，以上下文中与问题事件匹配的时间点为准（最细时间如「高三深秋」）；同一章内多处时间标记时，以紧贴该事件的时间为准。"
                        "回答时仅引用与问题直接相关的人物，不要罗列无关人物信息。"
                    )
                user_message = f"""基于以下文档内容回答用户问题：

{context}

用户问题：{user_query}

请基于上述文档内容回答问题。如果文档中没有相关信息，请诚实说明。{fact_check_hint}{structured_source_hint}"""
        else:
            user_message = user_query
        
        # 打印输入给LLM的 User Message（截断，便于排查）
        _max_log = 2500
        _user_preview = (user_message[: _max_log] + "\n...[截断 " + str(len(user_message) - _max_log) + " 字符]") if len(user_message) > _max_log else user_message
        logger.info("输入给LLM的 User Message（截断）：")
        for line in _user_preview.split("\n"):
            logger.info(line)
        
        messages.append({"role": "user", "content": user_message})
        
        # 调用LLM生成回答
        logger.info("正在调用LLM生成回答...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                stream=False
            )
            
            answer = response.choices[0].message.content
            
            logger.info("LLM回答生成完成\n")
            
            return {
                "answer": answer,
                "sources": sources if need_rag else [],
                "used_rag": need_rag,
                "rag_reason": rag_reason,
                "target_source": target_source,
                "question_type": question_type,
                "is_meta_analysis": is_meta_analysis
            }
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return {
                "answer": f"抱歉，处理您的问题时出现错误：{str(e)}",
                "sources": [],
                "used_rag": need_rag,
                "rag_reason": rag_reason,
                "target_source": None,
                "question_type": question_type if need_rag else None,
                "is_meta_analysis": False
            }
    
    def chat_stream(self, user_query: str, conversation_history: List[Dict] = None):
        """
        流式处理用户查询（用于实时显示）
        
        Yields:
            (chunk, is_final) - 回答片段和是否结束
        """
        import time
        # 记录各阶段开始时间
        total_start_time = time.time()
        route_start_time = time.time()
        
        conversation_history = self._normalize_and_trim_history(conversation_history, current_query=user_query)
        
        # 统一路由判断：同时完成"是否需要RAG"和"问题类型"的决策
        need_rag, question_type, type_reason, rag_reason = self._route_question(user_query)
        route_time = time.time() - route_start_time
        logger.info(f"[chat_stream] 路由判断: need_rag={need_rag}, question_type={question_type}")
        logger.info(f"  类型判断理由: {type_reason}")
        logger.info(f"  RAG判断理由: {rag_reason}")
        
        context = ""
        sources = []
        target_source = None
        rag_result = {}
        need_multi_hop = False  # 初始化多跳检索标识
        
        if need_rag and question_type != "none":
            # ========== 只对行为解释类和情节推演类问题使用Reasoning模块判断是否需要多跳检索 ==========
            # 判断是否为行为分析类问题（需要多跳推理）
            is_behavior_analysis_type = question_type in ["behavior_explanation", "scenario_simulation"]
            need_multi_hop = False  # 初始化多跳检索标识
            
            # 优化：如果是行为分析类或情节推演类问题，立即发送初始提示，让用户知道系统正在深度思考
            if is_behavior_analysis_type:
                initial_notification = "您的问题需要深度思考，请等待系统推理……"
                yield {"type": "multi_hop_notification", "message": initial_notification}, False
                logger.info(f"[多跳通知] 路由判断完成，立即发送初始提示: {initial_notification}")
            
            if is_behavior_analysis_type and self.reasoning_module:
                # 只对行为解释类和情节推演类问题使用reasoning模块
                try:
                    # 先进行一次初始检索，获取初始上下文用于reasoning判断
                    rewritten_query = self._rewrite_query(user_query)
                    initial_result = self._retrieve_context(rewritten_query, question_type, original_query=user_query)
                    initial_context = initial_result.get("context", "")
                    
                    # 调用reasoning模块判断当前推理步骤数量是否充足
                    reasoning_result = self.reasoning_module.reason(
                        user_query=user_query,
                        question_type=question_type,
                        initial_context=initial_context,
                        conversation_history=conversation_history
                    )
                    
                    need_multi_hop = reasoning_result.get("need_secondary_retrieval", False)
                    query_chain = reasoning_result.get("query_chain", [])
                    
                    logger.info(f"[chat_stream][行为分析类] Reasoning判断结果: need_multi_hop={need_multi_hop}")
                    logger.info(f"  推理说明: {reasoning_result.get('reasoning', '')}")
                    
                    if need_multi_hop and query_chain and len(query_chain) > 1:
                        # 使用多跳检索
                        logger.info("使用多跳检索模式")
                        logger.info(f"[多跳通知] 准备发送多跳检索通知，查询链长度: {len(query_chain)}")
                        # 注意：初始提示已在路由判断后发送，这里直接更新为带步骤数的提示
                        # 更新初始提示为带步骤数的版本
                        step_message = f"您的问题需要深度思考，正在执行深度思考推理（0/{len(query_chain)}）…"
                        yield {"type": "multi_hop_notification", "message": step_message}, False
                        logger.info(f"[多跳通知] 多跳检索步骤通知已发送: {step_message}")
                        
                        # 使用生成器包装多跳检索，以便在检索过程中发送进度更新
                        rag_result = None
                        for progress_update in self._multi_hop_retrieve_with_progress(query_chain, question_type, user_query):
                            if isinstance(progress_update, dict) and progress_update.get("type") == "multi_hop_progress":
                                # 发送进度更新
                                yield {"type": "multi_hop_notification", "message": progress_update.get("message")}, False
                            elif isinstance(progress_update, dict) and progress_update.get("type") == "multi_hop_result":
                                # 检索完成，获取结果
                                rag_result = progress_update.get("result")
                        
                        if not rag_result:
                            # 如果生成器没有返回结果，回退到普通方法
                            logger.warning("生成器版本多跳检索未返回结果，回退到普通方法")
                            rag_result = self._multi_hop_retrieve(query_chain, question_type, user_query)
                        context = rag_result.get("context", "")
                        sources = rag_result.get("sources", [])
                        target_source = rag_result.get("target_source", None)
                        # 标记为多跳检索
                        need_multi_hop = True
                    else:
                        # 使用单次检索（使用初始检索结果）
                        logger.info("使用单次检索模式（推理步骤充足）")
                        context = initial_result.get("context", "")
                        sources = initial_result.get("sources", [])
                        target_source = initial_result.get("target_source", None)
                        need_multi_hop = False
                        
                except Exception as e:
                    logger.error(f"Reasoning模块调用失败: {e}，回退到单次检索")
                    # 回退到单次检索
                    rewritten_query = self._rewrite_query(user_query)
                    rag_result = self._retrieve_context(rewritten_query, question_type, original_query=user_query)
                    context = rag_result.get("context", "")
                    sources = rag_result.get("sources", [])
                    target_source = rag_result.get("target_source", None)
                    need_multi_hop = False
            else:
                # 其他问题类型直接使用单次检索（不需要reasoning判断）
                logger.info(f"[chat_stream][{question_type}] 非行为分析类问题，直接使用单次检索")
                rewrite_start_time = time.time()
                rewritten_query = self._rewrite_query(user_query)
                rewrite_time = time.time() - rewrite_start_time
                rag_result = self._retrieve_context(rewritten_query, question_type, original_query=user_query)
                context = rag_result.get("context", "")
                sources = rag_result.get("sources", [])
                target_source = rag_result.get("target_source", None)
                need_multi_hop = False
                # 保存查询改写时间
                if isinstance(rag_result, dict):
                    rag_result["rewrite_time"] = rewrite_time
        
        # 判断是否为元文本分析问题
        is_meta_analysis = question_type == "meta_analysis" if need_rag and question_type else False
        
        # 判断是否为行为分析类问题
        is_behavior_analysis = question_type in ["behavior_explanation", "scenario_simulation"] if need_rag and question_type else False
        
        # 构建系统提示词（强调必须利用历史保持一致性）
        # 检查context中是否包含结构化数据源（用于chat_stream）
        has_structured_sources = False
        if context:
            structured_markers = [
                "[作者独家解读]",
                "[人物画像库 - 结构化信息]",
                "[时间线索引 - 用于事实核查]",
                "[关键事件链条 - 用于事实核查]",
                "[时间线:",
                "【1. 时间线索引】",
                "【3. 元文本分析库】",
                "【4. 相关人物心理模型】",
                "【6. 意象系统】"
            ]
            has_structured_sources = any(marker in context for marker in structured_markers)
        
        if is_meta_analysis:
            system_prompt = """你是一个得到作者真传的文学分析助手，专门回答关于《雪落成诗》和《影化成殇》这两部小说的创作意图、背景音乐、伏笔等元文本分析问题。

你的任务：
1. 严格依据作者提供的独家解读进行回答
2. 将作者解读作为核心依据，结合原文证据进行深入分析
3. 回答要深入、专业、有洞察力
4. 可以引用原文内容作为证据支撑

注意：
- 作者解读是权威上下文，必须优先使用
- 原文内容作为补充证据，用于支撑作者解读
- 回答要体现对创作技巧和意图的深度理解
- **信息来源规范（必须遵守）**：对用户呈现时，信息来源一律只写「原文」或「作品原文」。严禁在回答中出现时间线、人物画像、元文本知识库、关键事件链条、作者解读等任何内部数据源名称；只使用其内容作答，若需说明来源则仅写「原文」。
- **必须结合对话历史**：延续上下文、指代一致、不要把同一会话当作单轮问答"""
        elif is_behavior_analysis:
            # 统一的行为分析提示词（简化版；动机优先采信人物自述）
            system_prompt = """你是严谨的文学分析师，基于提供的上下文回答问题。

**动机采信原则**：优先采信人物自述的动机，其次再结合外部事件关联；人物自述动机高于外部推断。回答时按事实核查→模式分析→组织答案的流程进行。

注意：
- **信息来源规范（必须遵守）**：对用户呈现时，信息来源一律只写「原文」或「作品原文」。严禁在回答中出现时间线、人物画像、元文本分析、心理模型等任何内部数据源名称；只使用其内容作答，若需说明来源则仅写「原文」。
- 必须结合对话历史：延续上下文、指代一致"""
        else:
            system_prompt = f"""你是一个专业的小说问答助手，专门回答关于《雪落成诗》和《影化成殇》这两部小说的问题。

你的任务：
1. 基于提供的文档内容回答问题
2. 如果文档中没有相关信息，诚实地说不知道
3. 回答要准确、简洁、有依据
4. 可以适当引用原文内容，但要自然流畅

注意：
- 《雪落成诗》是主体小说
- 《影化成殇》是续集，分析时需要结合《雪落成诗》的背景，但只输出《影化成殇》中直接提及的内容
- **章节引用规范**：在引用章节时，请使用完整的章节名称，例如：
  * "第1卷 情感主线 第1章 缘起，云散"（而不是"1.1"或"第1卷第1章"）
  * "第2卷 人物篇 第1章 天台，隧道"（而不是"2.1"或"第2卷第1章"）
  * "第0卷 序章 雪落惊梦"（而不是"序章"或"0.1"）
- **信息来源规范（必须遵守）**：对用户呈现时，信息来源一律只写「原文」或「作品原文」。严禁在回答中出现时间线、人物画像、元文本知识库、关键事件链条等任何内部数据源名称；只使用其内容作答，若需说明来源则仅写「原文」。
- 保持回答的语言表述自然、流畅，符合中文阅读习惯
- **必须结合对话历史**：延续上下文、指代一致、不要把同一会话当作单轮问答"""

        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        
        if context:
            if is_meta_analysis:
                # 元文本分析的提示词格式
                # 分离作者解读和原文补充
                if '[来源:' in context:
                    parts = context.split('[来源:')
                    author_interpretation = parts[0].replace('[作者独家解读]', '').strip()
                    original_text = '\n\n[来源:'.join(parts[1:]) if len(parts) > 1 else ''
                else:
                    author_interpretation = context.replace('[作者独家解读]', '').strip() if '[作者独家解读]' in context else ''
                    original_text = context
                
                user_message = f"""你是一个得到作者真传的文学分析助手，请严格依据以下材料回答问题。

【作者独家解读】
{author_interpretation}

【作品原文补充】
{original_text}

问题：{user_query}

请以作者解读为核心，结合原文证据，给出深入的分析。

**信息来源规范**：回答中若需说明信息来源，一律只写「原文」或「作品原文」，不得出现元文本知识库、作者独家解读等任何内部名称。"""
            elif is_behavior_analysis:
                # 行为分析类问题的提示词格式（简化；动机优先采信人物自述）
                user_message = f"""基于以下上下文数据回答用户问题：

{context}

问题：{user_query}

请按照 system prompt 中的流程回答；分析动机时优先采信人物自述，再结合外部事件关联。

**信息来源规范**：回答中若需说明信息来源，一律只写「原文」或「作品原文」，不得出现时间线、人物画像、元文本分析库、意象系统、心理学概念库、情节推演规则等任何内部名称。"""
            else:
                structured_source_hint = ""
                if has_structured_sources:
                    structured_source_hint = "\n\n**信息来源规范**：回答中若需说明信息来源，一律只写「原文」或「作品原文」，不得出现任何内部数据源名称。"
                fact_check_hint = ""
                if "[时间线索引" in (context or "") or "时间线:" in (context or "") or "[关键事件链条" in (context or ""):
                    fact_check_hint = (
                        "\n\n**事实核查**："
                        "若问具体事件时间，以上下文中与问题事件匹配的时间点为准（最细时间如「高三深秋」）；同一章内多处时间标记时，以紧贴该事件的时间为准。"
                        "回答时仅引用与问题直接相关的人物，不要罗列无关人物信息。"
                    )
                user_message = f"""基于以下文档内容回答用户问题：

{context}

用户问题：{user_query}

请基于上述文档内容回答问题。如果文档中没有相关信息，请诚实说明。{fact_check_hint}{structured_source_hint}"""
        else:
            user_message = user_query
        
        # 记录最终输入给LLM的消息统计信息，并打印 User Message 内容（截断，便于排查）
        _max_log = 2500
        _user_preview = (user_message[: _max_log] + "\n...[截断 " + str(len(user_message) - _max_log) + " 字符]") if len(user_message) > _max_log else user_message
        logger.info("=" * 80)
        logger.info("准备输入给LLM的消息统计：")
        logger.info("=" * 80)
        logger.info(f"System Prompt长度: {len(system_prompt)} 字符")
        if conversation_history:
            logger.info(f"对话历史: {len(conversation_history)} 条消息")
        logger.info(f"User Message长度: {len(user_message)} 字符")
        logger.info(f"总消息数量: {len(messages) + 1} (包含system + 历史 + user)")
        logger.info(f"总字符数: {len(system_prompt) + sum(len(m.get('content', '')) for m in messages) + len(user_message)} 字符")
        logger.info("-" * 40)
        logger.info("输入给LLM的 User Message 内容（截断）：")
        logger.info("-" * 40)
        for line in _user_preview.split("\n"):
            logger.info(line)
        logger.info("=" * 80 + "\n")
        
        messages.append({"role": "user", "content": user_message})
        
        # 流式调用LLM
        import time
        llm_start_time = time.time()
        logger.info("正在流式调用LLM生成回答...")
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                stream=True
            )
            
            full_answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_answer += content
                    yield content, False
            
            llm_end_time = time.time()
            llm_elapsed_time = llm_end_time - llm_start_time
            logger.info(f"LLM流式回答生成完成，耗时: {llm_elapsed_time:.2f} 秒\n")
            
            # 计算总耗时
            total_time = time.time() - total_start_time
            
            # 收集所有时间统计
            if need_rag:
                if "rag_result" in locals() and isinstance(rag_result, dict):
                    final_rag_time = rag_result.get("rag_time", 0.0)
                    final_rewrite_time = rag_result.get("rewrite_time", 0.0)
                elif "initial_result" in locals() and isinstance(initial_result, dict):
                    final_rag_time = initial_result.get("rag_time", 0.0)
                    final_rewrite_time = initial_result.get("rewrite_time", 0.0)
                else:
                    final_rag_time = 0.0
                    final_rewrite_time = 0.0
            else:
                final_rag_time = 0.0
                final_rewrite_time = 0.0
            
            final_llm_time = llm_elapsed_time
            other_time = total_time - route_time - final_rewrite_time - final_rag_time - final_llm_time
            
            # 返回最终结果（包含时间统计）
            yield {
                "answer": full_answer,
                "sources": sources if need_rag else [],
                "used_rag": need_rag,
                "rag_reason": rag_reason,
                "target_source": target_source,
                "question_type": question_type,
                "is_meta_analysis": is_meta_analysis,
                "is_multi_hop": need_multi_hop if need_rag and question_type != "none" else False,
                "route_time": route_time,
                "rewrite_time": final_rewrite_time,
                "rag_time": final_rag_time,
                "llm_time": final_llm_time,
                "other_time": other_time,
                "total_time": total_time
            }, True
            
        except Exception as e:
            logger.error(f"LLM流式调用失败: {e}")
            yield {
                "answer": f"抱歉，处理您的问题时出现错误：{str(e)}",
                "sources": [],
                "used_rag": need_rag,
                "rag_reason": rag_reason,
                "target_source": None,
                "question_type": question_type if need_rag else None,
                "is_meta_analysis": False,
                "is_multi_hop": False
            }, True
