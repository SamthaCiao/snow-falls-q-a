"""
RAG检索器基类
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from rag_system import NovelRAGSystem


class BaseRetriever(ABC):
    """RAG检索器基类"""
    
    def __init__(self, rag_system: NovelRAGSystem):
        """
        初始化检索器
        
        Args:
            rag_system: RAG系统实例
        """
        self.rag_system = rag_system
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """
        检索相关上下文
        
        Args:
            query: 查询文本
            top_k: 返回top k个结果
        
        Returns:
            检索结果字典，包含：
            - context: 检索到的上下文文本
            - sources: 来源信息列表
            - score: 相关性分数
        """
        pass
