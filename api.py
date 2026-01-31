"""
FastAPI接口模块
提供RAG系统的API服务
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uvicorn
from rag_system import NovelRAGSystem
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="小说RAG系统API",
    description="基于LangChain和Sentence Transformers的小说检索问答系统",
    version="1.0.0"
)

# 全局RAG系统实例
rag_system: Optional[NovelRAGSystem] = None


class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str
    target_source: Optional[str] = None  # "雪落成诗" 或 "影化成殇"
    use_background: bool = True  # 是否使用背景信息
    top_k: int = 5


class AnswerResponse(BaseModel):
    """答案响应模型"""
    question: str
    answer: str
    sources: List[dict]
    target_source: str


class SearchResponse(BaseModel):
    """检索响应模型"""
    query: str
    target_source: str
    main_results: List[dict]
    sequel_results: List[dict]
    background_results: List[dict]


@app.on_event("startup")
async def startup_event():
    """启动时初始化RAG系统"""
    global rag_system
    from data_source_loader import get_data_source_dir
    os.environ["DATA_SOURCE_DIR"] = get_data_source_dir()
    print("正在初始化RAG系统...")
    
    try:
        rag_system = NovelRAGSystem(
            chunk_size=400,
            chunk_overlap=50
        )
        
        # 构建索引(如果不存在)
        print("正在检查索引...")
        rag_system.build_index(force_rebuild=False)
        print("RAG系统初始化完成!")
    except Exception as e:
        print(f"初始化失败: {e}")
        print("\n请检查:")
        print("1. 网络连接是否正常")
        print("2. 是否已安装所有依赖")
        print("3. 数据源文件是否存在")
        # 不抛出异常,让服务启动但返回错误信息
        rag_system = None


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "小说RAG系统API",
        "version": "1.0.0",
        "endpoints": {
            "answer": "/api/answer - 回答问题",
            "search": "/api/search - 检索文档",
            "rebuild_index": "/api/rebuild_index - 重建索引",
            "docs": "/docs - Swagger UI文档"
        }
    }


@app.post("/api/answer", response_model=AnswerResponse)
async def answer_question(request: QueryRequest):
    """
    回答问题
    
    - **question**: 用户问题
    - **target_source**: 目标来源("雪落成诗"或"影化成殇"), None表示自动判断
    - **use_background**: 是否使用背景信息(当查询《影化成殇》时结合《雪落成诗》背景)
    - **top_k**: 返回的top k个结果
    """
    if rag_system is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG系统未初始化。请检查模型是否已下载,或运行 python setup_model.py 手动下载模型"
        )
    
    # 记录开始时间
    start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"[用户提问] {request.question[:100]}{'...' if len(request.question) > 100 else ''}")
    logger.info(f"[开始时间] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    try:
        result = rag_system.answer_question(
            question=request.question,
            target_source=request.target_source,
            use_background=request.use_background
        )
        
        # 计算并记录总耗时
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"[完成时间] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        logger.info(f"[总耗时] {elapsed_time:.2f} 秒")
        logger.info("=" * 80)
        
        return AnswerResponse(**result)
    except Exception as e:
        # 即使出错也记录耗时
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.error(f"[错误] 处理问题时出错: {str(e)}")
        logger.info(f"[完成时间] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        logger.info(f"[总耗时] {elapsed_time:.2f} 秒")
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=f"处理问题时出错: {str(e)}")


@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: QueryRequest):
    """
    检索文档
    
    - **question**: 查询文本
    - **target_source**: 目标来源
    - **use_background**: 是否使用背景信息
    - **top_k**: 返回的top k个结果
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG系统未初始化")
    
    try:
        result = rag_system.search(
            query=request.question,
            target_source=request.target_source,
            top_k=request.top_k,
            use_background=request.use_background,
            use_hybrid=True  # 明确启用混合检索
        )
        return SearchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索时出错: {str(e)}")


@app.post("/api/rebuild_index")
async def rebuild_index():
    """
    重建索引(谨慎使用,耗时较长)
    
    注意: 如果检索结果不准确,可能需要重建索引
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG系统未初始化")
    
    try:
        print("开始重建索引...")
        rag_system.build_index(force_rebuild=True)
        return {
            "message": "索引重建完成",
            "note": "索引已更新,现在可以使用新的检索逻辑"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建索引时出错: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
