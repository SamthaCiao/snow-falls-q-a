"""
Reasoning模块 - 使用DeepSeek Reasoner进行推理判断
判断是否需要二次检索，并生成链式查询意图
"""
import os
import json
import logging
import requests
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ReasoningModule:
    """推理模块 - 使用DeepSeek Reasoner进行多跳推理判断"""
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 120):
        """
        初始化推理模块
        
        Args:
            api_key: DeepSeek API密钥，如果为None则从环境变量读取
            timeout: API调用超时时间（秒），默认120秒，适应thinking mode的推理时间
        """
        # 加载环境变量
        load_dotenv('SUPER_MIND_API_KEY.env')
        load_dotenv('.env')
        
        # 读取API密钥
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "未找到DEEPSEEK_API_KEY！\n"
                "请在以下位置之一设置API密钥：\n"
                "1. SUPER_MIND_API_KEY.env 文件中设置 DEEPSEEK_API_KEY\n"
                "2. .env 文件中设置 DEEPSEEK_API_KEY\n"
                "3. 环境变量中设置 DEEPSEEK_API_KEY"
            )
        
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.timeout = timeout
        logger.info(f"Reasoning模块初始化完成，超时时间: {timeout}秒")
    
    def reason(self, 
               user_query: str, 
               question_type: str,
               initial_context: Optional[str] = None,
               conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        使用DeepSeek Reasoner进行推理，判断是否需要二次检索并生成查询链
        
        Args:
            user_query: 用户原始查询
            question_type: 问题类型（从路由判断获得）
            initial_context: 初始检索到的上下文（可选，用于判断是否需要二次检索）
            conversation_history: 对话历史（可选）
        
        Returns:
            推理结果字典，包含：
            - need_secondary_retrieval: 是否需要二次检索
            - query_chain: 查询链（链式结构），格式: [{"query": "...", "intent": "...", "step": 1}, ...]
            - reasoning: 推理过程说明
        """
        logger.info("=" * 60)
        logger.info("开始Reasoning推理...")
        logger.info(f"用户查询: {user_query}")
        logger.info(f"问题类型: {question_type}")
        logger.info("=" * 60)
        
        # 构建系统提示词（优化为更简洁的版本，减少token消耗）
        system_prompt = """分析用户问题，判断是否需要多跳检索。

任务：
1. 判断是否需要二次检索
2. 如需要，生成查询链

需要二次检索的情况：
- 多实体关系（如"A和B的关系"）
- 需要背景信息（如"为什么X这样做"需先查X的行为）
- 推理链条（如"如果A发生，B会怎样"）
- 对比分析（如"A和B的区别"）

不需要二次检索：
- 可直接回答（如"X是谁"、"X做了什么"）
- 上下文已足够

返回JSON：
{
  "need_secondary_retrieval": true/false,
  "reasoning": "推理说明",
  "query_chain": [
    {"step": 1, "query": "查询文本", "intent": "意图", "depends_on": null},
    {"step": 2, "query": "查询文本", "intent": "意图", "depends_on": 1}
  ]
}
如need_secondary_retrieval=false，query_chain只包含一个元素。"""
        
        # 构建用户消息
        user_message_parts = [f"用户问题：{user_query}"]
        user_message_parts.append(f"问题类型：{question_type}")
        
        if initial_context:
            # 限制初始上下文长度，避免请求过大导致超时
            context_preview = initial_context[:300]  # 减少到300字符
            user_message_parts.append(f"\n初始检索到的上下文（前300字符）：\n{context_preview}...")
            user_message_parts.append("\n请基于以上信息判断是否需要二次检索。")
        else:
            user_message_parts.append("\n请分析问题并判断是否需要二次检索。")
        
        if conversation_history:
            user_message_parts.append("\n对话历史（最近2轮）：")
            for msg in conversation_history[-4:]:  # 减少到最近2轮（每轮2条消息）
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:150]  # 进一步限制长度
                user_message_parts.append(f"{role}: {content}")
        
        user_message = "\n".join(user_message_parts)
        
        try:
            # 调用DeepSeek Reasoner API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "stream": False,
                "temperature": 0.3
            }
            
            logger.info("正在调用DeepSeek Reasoner API...")
            logger.info(f"请求URL: {self.api_url}")
            logger.info(f"模型: {payload['model']}")
            logger.info(f"消息长度: {len(user_message)} 字符")
            
            # DeepSeek Reasoner是thinking mode，需要更长的响应时间
            # 根据文档，thinking mode可能需要60-120秒甚至更长时间
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"DeepSeek Reasoner API调用失败: HTTP {response.status_code}")
                logger.error(f"响应内容: {response.text[:500]}")
                # 尝试解析错误信息
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_info = error_data['error']
                        logger.error(f"API错误: {error_info.get('message', '未知错误')}")
                        logger.error(f"错误类型: {error_info.get('type', 'unknown')}")
                except:
                    pass
                # 返回默认结果（不需要二次检索）
                return self._default_result(user_query)
            
            result = response.json()
            
            # 检查是否有错误
            if 'error' in result:
                error_msg = result.get('error', {}).get('message', '未知错误')
                logger.error(f"DeepSeek Reasoner API返回错误: {error_msg}")
                return self._default_result(user_query)
            
            # 解析响应
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                
                # 处理可能被包裹在代码块中的JSON
                content_clean = content.strip()
                if content_clean.startswith('```json'):
                    content_clean = content_clean.replace('```json', '').replace('```', '').strip()
                elif content_clean.startswith('```'):
                    content_clean = content_clean.replace('```', '').strip()
                
                try:
                    parsed = json.loads(content_clean)
                    
                    # 验证和规范化结果
                    need_secondary = parsed.get("need_secondary_retrieval", False)
                    reasoning = parsed.get("reasoning", "推理过程")
                    query_chain = parsed.get("query_chain", [])
                    
                    # 验证query_chain格式
                    if not query_chain or not isinstance(query_chain, list):
                        logger.warning("query_chain格式无效，使用默认结果")
                        return self._default_result(user_query)
                    
                    # 确保query_chain至少包含一个元素
                    if len(query_chain) == 0:
                        query_chain = [{
                            "step": 1,
                            "query": user_query,
                            "intent": "原始查询",
                            "depends_on": None
                        }]
                    
                    logger.info(f"Reasoning推理完成: need_secondary_retrieval={need_secondary}")
                    logger.info(f"  查询链长度: {len(query_chain)}")
                    for i, qc in enumerate(query_chain, 1):
                        logger.info(f"  步骤{i}: {qc.get('query', '')[:50]}...")
                    
                    return {
                        "need_secondary_retrieval": need_secondary,
                        "reasoning": reasoning,
                        "query_chain": query_chain
                    }
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败: {e}")
                    logger.error(f"响应内容: {content_clean[:500]}")
                    return self._default_result(user_query)
            else:
                logger.error(f"DeepSeek Reasoner API响应格式异常: {result}")
                return self._default_result(user_query)
                
        except requests.exceptions.Timeout:
            logger.error(f"DeepSeek Reasoner API调用超时（{self.timeout}秒）")
            logger.warning("Reasoning模式（thinking mode）需要较长推理时间")
            logger.warning("建议：1) 增加超时时间 2) 优化提示词长度 3) 减少输入上下文")
            return self._default_result(user_query)
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek Reasoner API网络请求失败: {e}")
            logger.error(f"错误类型: {type(e).__name__}")
            return self._default_result(user_query)
        except Exception as e:
            logger.error(f"DeepSeek Reasoner API调用失败: {e}")
            logger.error(f"错误类型: {type(e).__name__}")
            import traceback
            logger.debug(f"详细错误信息: {traceback.format_exc()}")
            return self._default_result(user_query)
    
    def _default_result(self, user_query: str) -> Dict:
        """
        返回默认结果（不需要二次检索）
        
        Args:
            user_query: 用户查询
        
        Returns:
            默认推理结果
        """
        return {
            "need_secondary_retrieval": False,
            "reasoning": "推理失败，使用默认单次检索",
            "query_chain": [{
                "step": 1,
                "query": user_query,
                "intent": "原始查询",
                "depends_on": None
            }]
        }
