"""
基于FastAPI的Web聊天界面
参考ChatGPT风格设计
部署到 GitHub Pages 时：前端设 window.__API_BASE__ 为本服务地址，本端设 CORS_ORIGINS 放行前端域名。
"""
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import json
import asyncio
import os
from pathlib import Path
from llm_rag_system import LLMRAGSystem
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

app = FastAPI(title="小说RAG智能问答系统")

# CORS：部署到 GitHub Pages 时设置 CORS_ORIGINS（逗号分隔），例如 https://username.github.io
_cors_origins = os.environ.get("CORS_ORIGINS", "").strip()
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in _cors_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

# 挂载静态文件目录
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 全局RAG系统实例
llm_rag_system: Optional[LLMRAGSystem] = None


class ChatMessage(BaseModel):
    role: str
    content: str
    sources: Optional[List[dict]] = None
    used_rag: Optional[bool] = False


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = None


class RouteCheckRequest(BaseModel):
    message: str


# ---------- 共享历史（内存存储，不写盘；重启后清空）----------
# 所有用户、所有设备可见同一份列表；无需登录
_SHARED_HISTORY: List[dict] = []
_SHARED_HISTORY_MAX = 200


class SharedConversationItem(BaseModel):
    """单条共享对话：id、标题、时间、完整对话树与当前路径"""
    id: str
    title: str
    timestamp: str
    tree: dict
    current_path: List[str]


@app.get("/api/shared_history")
async def get_shared_history():
    """返回全部用户共享的对话历史（内存中的列表，含完整 tree/current_path 便于前端直接加载）"""
    return JSONResponse(content={"items": _SHARED_HISTORY.copy()})


@app.post("/api/shared_history")
async def post_shared_history(item: SharedConversationItem):
    """上报一条对话到共享历史（合并：同 id 则更新，否则追加；超过上限则去掉最旧的）"""
    global _SHARED_HISTORY
    payload = item.model_dump()
    existing = next((i for i, x in enumerate(_SHARED_HISTORY) if x.get("id") == item.id), None)
    if existing is not None:
        _SHARED_HISTORY[existing] = payload
    else:
        _SHARED_HISTORY.insert(0, payload)
    while len(_SHARED_HISTORY) > _SHARED_HISTORY_MAX:
        _SHARED_HISTORY.pop()
    return JSONResponse(content={"ok": True, "count": len(_SHARED_HISTORY)})


@app.on_event("startup")
async def startup_event():
    """启动时初始化RAG系统"""
    global llm_rag_system
    from data_source_loader import get_data_source_dir
    os.environ["DATA_SOURCE_DIR"] = get_data_source_dir()
    logger.info("=" * 80)
    logger.info("FastAPI服务启动，开始初始化LLM+RAG系统...")
    logger.info("=" * 80)
    try:
        llm_rag_system = LLMRAGSystem()
        logger.info("=" * 80)
        logger.info("✓ FastAPI服务初始化完成！")
        logger.info("  访问地址: http://localhost:8000")
        logger.info("  API文档: http://localhost:8000/docs")
        logger.info("=" * 80)
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"✗ 系统初始化失败: {e}")
        logger.error("=" * 80)
        llm_rag_system = None


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回主页面"""
    # 读取HTML模板文件
    template_path = Path(__file__).parent / "templates" / "chat.html"
    if template_path.exists():
        with open(template_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        # 如果模板文件不存在，返回简单的错误提示
        return HTMLResponse(
            content="<h1>错误：模板文件未找到</h1><p>请确保 templates/chat.html 文件存在</p>",
            status_code=500
        )


@app.post("/api/route-check")
async def route_check(request: RouteCheckRequest):
    """快速路由检查，用于前端提前显示提示"""
    if llm_rag_system is None:
        return JSONResponse(
            status_code=503,
            content={"error": "系统未初始化", "is_meta_analysis": False}
        )
    
    try:
        # 进行路由判断
        need_rag, question_type, type_reason, rag_reason = llm_rag_system._route_question(request.message)
        is_meta_analysis = (question_type == "meta_analysis") if need_rag and question_type else False
        
        logger.info(f"[route-check] 快速路由检查: question_type={question_type}, is_meta_analysis={is_meta_analysis}")
        logger.info(f"  类型判断理由: {type_reason}")
        logger.info(f"  RAG判断理由: {rag_reason}")
        logger.info(f"  路由判断结果已发送给前端UI模块")
        
        return JSONResponse(content={
            "is_meta_analysis": is_meta_analysis,
            "question_type": question_type,
            "need_rag": need_rag
        })
    except Exception as e:
        logger.error(f"路由检查失败: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "is_meta_analysis": False}
        )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """处理聊天请求（流式响应）"""
    if llm_rag_system is None:
        return JSONResponse(
            status_code=503,
            content={"error": "系统未初始化"}
        )
    
    # 记录开始时间
    start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"[用户提问] {request.message[:100]}{'...' if len(request.message) > 100 else ''}")
    logger.info(f"[开始时间] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    async def generate_stream():
        """生成流式响应"""
        try:
            result = None
            full_answer = ""
            has_sent_final = False
            
            # 关键：将同步生成器转换为异步迭代，确保每次yield后都能立即发送
            import asyncio
            from collections import deque
            
            # 创建一个队列来缓冲同步生成器的输出
            chunk_queue = deque()
            stream_done = False
            
            # 在后台线程中运行同步生成器
            def run_sync_generator():
                nonlocal stream_done
                try:
                    for chunk, is_final in llm_rag_system.chat_stream(
                        request.message,
                        request.conversation_history or []
                    ):
                        chunk_queue.append((chunk, is_final))
                        # 如果是多跳通知，立即标记，让主循环优先处理
                        if isinstance(chunk, dict) and chunk.get("type") == "multi_hop_notification":
                            from datetime import datetime
                            send_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                            logger.info(f"[多跳通知][后台线程][{send_time}] 检测到多跳检索，已加入队列: {chunk.get('message', '')}")
                finally:
                    stream_done = True
            
            # 在后台线程中启动同步生成器
            import threading
            sync_thread = threading.Thread(target=run_sync_generator, daemon=True)
            sync_thread.start()
            
            # 主循环：从队列中取出数据并yield
            while not stream_done or chunk_queue:
                if chunk_queue:
                    chunk, is_final = chunk_queue.popleft()
                    
                    if is_final:
                        # 最终结果
                        result = chunk
                        has_sent_final = True
                        # 发送最终结果
                        try:
                            yield f"data: {json.dumps({'type': 'final', 'data': result}, ensure_ascii=False)}\n\n"
                        except Exception as e:
                            logger.error(f"发送final消息时出错: {e}")
                            yield f"data: {json.dumps({'type': 'error', 'error': f'发送结果时出错: {str(e)}'}, ensure_ascii=False)}\n\n"
                    else:
                        # 检查是否为通知消息（字典类型）
                        if isinstance(chunk, dict):
                            # 多跳检索通知
                            if chunk.get("type") == "multi_hop_notification":
                                try:
                                    from datetime import datetime
                                    send_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                                    logger.info(f"[多跳通知][主循环][{send_time}] 从队列取出，准备发送给前端: {chunk.get('message', '')}")
                                    # 确保立即刷新缓冲区，避免SSE数据被缓冲
                                    notification_data = f"data: {json.dumps({'type': 'multi_hop_notification', 'message': chunk.get('message', '您的问题相对复杂，正在执行深度思考推理…')}, ensure_ascii=False)}\n\n"
                                    yield notification_data
                                    # 关键：在异步生成器中，使用 await asyncio.sleep(0) 确保事件循环有机会发送数据
                                    await asyncio.sleep(0)
                                    logger.info(f"[多跳通知][主循环][{send_time}] 通知已yield，等待发送")
                                except Exception as e:
                                    logger.error(f"发送多跳通知时出错: {e}")
                            # 其他类型的通知也可以在这里处理
                            continue
                        # 流式文本chunk
                        if isinstance(chunk, str) and chunk:
                            full_answer += chunk
                            # 立即发送chunk，不等待
                            try:
                                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"
                            except Exception as e:
                                logger.error(f"发送chunk时出错: {e}")
                                # chunk发送失败不影响整体流程，继续
                else:
                    # 队列为空，等待一小段时间再检查
                    await asyncio.sleep(0.01)
            
            if not has_sent_final:
                # 如果没有流式结果，使用非流式方法
                try:
                    result = llm_rag_system.chat(
                        request.message,
                        request.conversation_history or []
                    )
                    yield f"data: {json.dumps({'type': 'final', 'data': result}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    logger.error(f"非流式调用失败: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'error': f'处理请求时出错: {str(e)}'}, ensure_ascii=False)}\n\n"
            
            # 计算并记录总耗时
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 提取详细的时间统计
            route_time = result.get("route_time", 0.0) if result else 0.0
            rewrite_time = result.get("rewrite_time", 0.0) if result else 0.0
            rag_time = result.get("rag_time", 0.0) if result else 0.0
            llm_time = result.get("llm_time", 0.0) if result else 0.0
            other_time = result.get("other_time", 0.0) if result else 0.0
            total_time = result.get("total_time", elapsed_time) if result else elapsed_time
            used_rag = result.get("used_rag", False) if result else False
            
            logger.info(f"[完成时间] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
            logger.info("=" * 80)
            logger.info("[耗时统计详情]")
            logger.info(f"  路由判断用时: {route_time:.2f} 秒")
            if rewrite_time > 0.01:
                logger.info(f"  查询改写用时: {rewrite_time:.2f} 秒")
            if used_rag:
                logger.info(f"  RAG查询用时: {rag_time:.2f} 秒")
            logger.info(f"  LLM生成回答用时: {llm_time:.2f} 秒")
            if other_time > 0.1:
                logger.info(f"  其他处理用时: {other_time:.2f} 秒")
            logger.info("-" * 80)
            logger.info(f"[耗时总结] 总耗时: {total_time:.2f} 秒")
            if used_rag:
                logger.info(f"  - RAG相关: {rewrite_time + rag_time:.2f} 秒 ({((rewrite_time + rag_time) / total_time * 100):.1f}%)")
            logger.info(f"  - LLM生成: {llm_time:.2f} 秒 ({(llm_time / total_time * 100):.1f}%)")
            logger.info(f"  - 其他处理: {route_time + other_time:.2f} 秒 ({((route_time + other_time) / total_time * 100):.1f}%)")
            logger.info("=" * 80)
            
        except Exception as e:
            # 即使出错也记录耗时
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.error(f"[错误] 处理聊天请求时出错: {e}")
            logger.info(f"[完成时间] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
            logger.info(f"[总耗时] {elapsed_time:.2f} 秒")
            logger.info("=" * 80)
            yield f"data: {json.dumps({'type': 'error', 'error': f'处理请求时出错: {str(e)}'}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点，用于真正的流式输出"""
    await websocket.accept()
    
    if llm_rag_system is None:
        await websocket.send_json({"error": "系统未初始化"})
        await websocket.close()
        return
    
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            conversation_history = data.get("conversation_history", [])
            
            if not message:
                continue
            
            # 记录开始时间
            start_time = time.time()
            logger.info("=" * 80)
            logger.info(f"[WebSocket][用户提问] {message[:100]}{'...' if len(message) > 100 else ''}")
            logger.info(f"[开始时间] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
            
            # 先进行路由判断，检测是否为元文本分析（第一次路由判断）
            is_meta_analysis = False
            try:
                need_rag, question_type, type_reason, rag_reason = llm_rag_system._route_question(message)
                is_meta_analysis = (question_type == "meta_analysis") if need_rag and question_type else False
                logger.info(f"[WebSocket] 路由判断结果: need_rag={need_rag}, question_type={question_type}, is_meta_analysis={is_meta_analysis}")
                logger.info(f"  类型判断理由: {type_reason}")
                logger.info(f"  RAG判断理由: {rag_reason}")
                logger.info(f"  路由判断结果已发送给WebSocket客户端")
                
                # 如果检测到元文本分析，立即发送通知
                if is_meta_analysis:
                    await websocket.send_json({
                        "type": "meta_analysis_notification",
                        "message": "您已进入全新视角，正在调用元文本分析知识库…"
                    })
            except Exception as e:
                logger.error(f"WebSocket路由判断失败: {e}")
            
            # 流式处理
            for chunk, is_final in llm_rag_system.chat_stream(message, conversation_history):
                if is_final:
                    # 计算并记录总耗时
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    # 提取详细的时间统计
                    route_time = chunk.get("route_time", 0.0) if isinstance(chunk, dict) else 0.0
                    rewrite_time = chunk.get("rewrite_time", 0.0) if isinstance(chunk, dict) else 0.0
                    rag_time = chunk.get("rag_time", 0.0) if isinstance(chunk, dict) else 0.0
                    llm_time = chunk.get("llm_time", 0.0) if isinstance(chunk, dict) else 0.0
                    other_time = chunk.get("other_time", 0.0) if isinstance(chunk, dict) else 0.0
                    total_time = chunk.get("total_time", elapsed_time) if isinstance(chunk, dict) else elapsed_time
                    used_rag = chunk.get("used_rag", False) if isinstance(chunk, dict) else False
                    
                    logger.info(f"[完成时间] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
                    logger.info("=" * 80)
                    logger.info("[耗时统计详情]")
                    logger.info(f"  路由判断用时: {route_time:.2f} 秒")
                    if rewrite_time > 0.01:
                        logger.info(f"  查询改写用时: {rewrite_time:.2f} 秒")
                    if used_rag:
                        logger.info(f"  RAG查询用时: {rag_time:.2f} 秒")
                    logger.info(f"  LLM生成回答用时: {llm_time:.2f} 秒")
                    if other_time > 0.1:
                        logger.info(f"  其他处理用时: {other_time:.2f} 秒")
                    logger.info("-" * 80)
                    logger.info(f"[耗时总结] 总耗时: {total_time:.2f} 秒")
                    if used_rag:
                        logger.info(f"  - RAG相关: {rewrite_time + rag_time:.2f} 秒 ({((rewrite_time + rag_time) / total_time * 100):.1f}%)")
                    logger.info(f"  - LLM生成: {llm_time:.2f} 秒 ({(llm_time / total_time * 100):.1f}%)")
                    logger.info(f"  - 其他处理: {route_time + other_time:.2f} 秒 ({((route_time + other_time) / total_time * 100):.1f}%)")
                    logger.info("=" * 80)
                    
                    await websocket.send_json({
                        "type": "final",
                        "data": chunk,
                        "is_meta_analysis": chunk.get("is_meta_analysis", False) or is_meta_analysis if isinstance(chunk, dict) else is_meta_analysis
                    })
                else:
                    # 兼容：llm_rag_system.chat_stream 可能在中间过程 yield dict 通知（如多跳检索进度）
                    if isinstance(chunk, dict):
                        # 多跳检索通知（用于替代UI里三个点等待动画）
                        if chunk.get("type") == "multi_hop_notification":
                            await websocket.send_json({
                                "type": "multi_hop_notification",
                                "message": chunk.get("message", "您的问题相对复杂，正在执行深度思考推理…")
                            })
                            continue
                        # 未来可扩展：其他通知类型
                        await websocket.send_json({
                            "type": chunk.get("type", "notification"),
                            "data": chunk
                        })
                        continue

                    # 普通流式文本chunk
                    await websocket.send_json({
                        "type": "chunk",
                        "data": chunk
                    })
                    
    except WebSocketDisconnect:
        logger.info("WebSocket连接断开")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        await websocket.send_json({"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
