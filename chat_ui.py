"""
聊天式UI界面 - 参考ChatGPT设计
使用Streamlit构建
"""
import streamlit as st
import sys
from llm_rag_system import LLMRAGSystem
import logging
import time

# 配置日志（如果还没有配置）
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
logger = logging.getLogger(__name__)

# 配置页面
st.set_page_config(
    page_title="小说RAG智能问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .user-avatar {
        background-color: #1976d2;
        color: white;
    }
    .assistant-avatar {
        background-color: #424242;
        color: white;
    }
    .source-info {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #ddd;
    }
    .stButton>button {
        width: 100%;
        background-color: #1976d2;
        color: white;
        font-weight: bold;
    }
    .meta-analysis-notification {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: rgba(33, 150, 243, 0.95);
        color: white;
        padding: 2rem 3rem;
        border-radius: 1rem;
        font-size: 1.2rem;
        font-weight: bold;
        z-index: 9999;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translate(-50%, -60%); }
        to { opacity: 1; transform: translate(-50%, -50%); }
    }
    .system-agreement-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: rgba(44, 62, 80, 0.7);
        z-index: 9998;
        animation: fadeInOverlay 0.3s ease-in;
    }
    .system-agreement-modal-wrapper {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 9999;
        background-color: #f8f9fa;
        border-radius: 1rem;
        padding: 2.5rem 3rem;
        max-width: 700px;
        width: 90%;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        border: 1px solid #bdc3c7;
        animation: fadeInModal 0.3s ease-in;
    }
    .system-agreement-button-wrapper {
        text-align: center;
        margin-top: 1.5rem;
        padding-top: 1rem;
    }
    /* 弹窗确认按钮样式 */
    button[data-testid*="agreement_confirm"] {
        background: linear-gradient(135deg, #5d8aa8 0%, #2c3e50 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    button[data-testid*="agreement_confirm"]:hover {
        background: linear-gradient(135deg, #6b9bc0 0%, #34495e 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    .system-agreement-title {
        font-family: 'Georgia', 'Times New Roman', serif;
        font-size: 1.8rem;
        font-weight: 300;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        text-align: center;
        letter-spacing: 0.05em;
    }
    .system-agreement-content {
        font-size: 1rem;
        line-height: 1.8;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .system-agreement-content p {
        margin-bottom: 1rem;
    }
    .system-agreement-content strong {
        color: #2c3e50;
        font-weight: 600;
    }
    @keyframes fadeInOverlay {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeInModal {
        from { opacity: 0; transform: translate(-50%, -60%) scale(0.9); }
        to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_rag_system" not in st.session_state:
    with st.spinner("正在初始化系统，请稍候..."):
        try:
            st.session_state.llm_rag_system = LLMRAGSystem()
            st.success("系统初始化完成！")
        except Exception as e:
            st.error(f"系统初始化失败: {str(e)}")
            st.stop()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# 检查用户是否已确认系统说明
if "agreement_confirmed" not in st.session_state:
    st.session_state.agreement_confirmed = False

# 系统说明弹窗（如果用户未确认）
if not st.session_state.agreement_confirmed:
    # 创建遮罩层和弹窗
    st.markdown("""
    <div class="system-agreement-overlay"></div>
    <div class="system-agreement-modal-wrapper">
        <div class="system-agreement-title">📚 关于这个系统</div>
        <div class="system-agreement-content">
            <p>这是一个基于<strong>《雪落成诗》</strong>与<strong>《影化成殇》</strong>构建的对话式阅读系统，为读者提供一种多重自我共时叙事。</p>
            <p>它并非用于提供标准答案，也不试图替代你的阅读与判断。</p>
            <p>你可以向它询问情节、人物、时间线、意象，<br>
            也可以向它提出更开放的问题——<br>
            关于结构、重复、断裂、情绪的变化，以及文本未明言之处。</p>
            <p><strong>请注意：</strong><br>
            它的回答并非作者的最终解释，<br>
            而是一种基于文本结构与元信息的推演结果。</p>
            <p>你可以同意、质疑，或完全否定它的判断。</p>
            <p>这不是一次问答，<br>
            而是一种对话式的阅读方式。</p>
        </div>
        <div class="system-agreement-button-wrapper" id="agreement-button-container"></div>
    </div>
    <script>
        (function() {
            function moveButton() {
                var button = document.querySelector('button[data-testid*="agreement_confirm"]');
                var container = document.getElementById('agreement-button-container');
                if (button && container && container.children.length === 0) {
                    var buttonParent = button.parentElement;
                    if (buttonParent) {
                        container.appendChild(button);
                        button.style.width = '100%';
                        button.style.padding = '0.8rem 2rem';
                        button.style.fontSize = '1.1rem';
                        button.style.fontWeight = '500';
                    }
                }
            }
            // 立即尝试
            moveButton();
            // 监听 DOM 变化
            var observer = new MutationObserver(moveButton);
            observer.observe(document.body, { childList: true, subtree: true });
            // 延迟执行
            setTimeout(moveButton, 100);
            setTimeout(moveButton, 500);
        })();
    </script>
    """, unsafe_allow_html=True)
    
    # 确认按钮（会被 JavaScript 移动到弹窗内）
    if st.button("✅ 确认，我已知悉", use_container_width=True, type="primary", key="agreement_confirm"):
        st.session_state.agreement_confirmed = True
        st.rerun()
    
    # 如果未确认，不显示主界面内容
    st.stop()

# 标题（只有在用户确认后才显示）
st.markdown('<div class="main-header">📚 小说RAG智能问答系统</div>', unsafe_allow_html=True)
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    
    # 显示系统信息
    st.subheader("系统信息")
    st.info("""
    **功能说明：**
    - 智能判断是否需要检索文档
    - 基于小说内容回答问题
    - 支持多轮对话
    
    **数据源：**
    - 《雪落成诗》
    - 《影化成殇》
    """)
    
    # 清空对话按钮
    if st.button("🗑️ 清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()
    
    # 显示统计信息
    st.subheader("📊 统计")
    st.metric("对话轮数", len(st.session_state.messages) // 2)
    st.metric("RAG调用次数", sum(1 for msg in st.session_state.messages 
                                  if msg.get("role") == "assistant" and msg.get("used_rag", False)))
    
    # 调试信息（用于排查问题）
    if "debug_meta_analysis" in st.session_state:
        with st.expander("🔍 调试信息（路由判断结果）", expanded=False):
            st.text(st.session_state.debug_meta_analysis)

# 主聊天区域
chat_container = st.container()

# 显示历史消息
with chat_container:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        used_rag = message.get("used_rag", False)
        sources = message.get("sources", [])
        
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        else:
            with st.chat_message("assistant"):
                st.write(content)
                
                # 显示RAG信息
                if used_rag:
                    with st.expander("📖 参考来源", expanded=False):
                        if sources:
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**来源 {i}:** {source.get('chapter', '未知章节')}")
                                st.caption(source.get('content', '')[:200] + "...")
                        else:
                            st.info("未找到相关文档")
                else:
                    st.caption("💬 直接回答（未使用RAG）")

# 用户输入
user_input = st.chat_input("请输入您的问题...")

if user_input:
    # 记录开始时间
    start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"[用户提问] {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
    logger.info(f"[开始时间] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    # 添加用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    # 显示用户消息
    with st.chat_message("user"):
        st.write(user_input)
    
    # 在流式处理前先进行路由判断，检测是否为元文本分析（第一次路由判断）
    is_meta_analysis = False
    question_type = None
    try:
        need_rag, question_type, type_reason, rag_reason = st.session_state.llm_rag_system._route_question(user_input)
        is_meta_analysis = (question_type == "meta_analysis" )
        # 调试信息（保存到session state用于后续显示）
        st.session_state.debug_meta_analysis = f"路由判断结果: need_rag={need_rag}, question_type={question_type}, is_meta_analysis={is_meta_analysis}"
    except Exception as e:
        # 如果路由判断失败，记录错误并继续正常流程
        logger.error(f"路由判断失败: {e}")
        st.session_state.debug_meta_analysis = f"路由判断失败: {e}"
    
    # 生成回答
    with st.chat_message("assistant"):
        # 如果是元文本分析，立即显示提示信息（在RAG调用之前）
        meta_notification_placeholder = None
        if is_meta_analysis:
            # 使用醒目的提示框
            meta_notification_placeholder = st.empty()
            # 直接使用warning，确保在RAG调用前显示
            meta_notification_placeholder.warning("🌟 **您已进入全新视角，正在调用元文本分析知识库…**")
        
        message_placeholder = st.empty()
        full_response = ""
        used_rag = False
        sources = []
        rag_reason = ""
        is_multi_hop = False
        multi_hop_notification_placeholder = None
        
        try:
            # 调用chat_stream（内部会进行第二次路由判断和RAG调用）
            # 注意：提示已经在上面显示，会保持显示直到收到第一个流式chunk
            first_chunk_received = False
            for chunk, is_final in st.session_state.llm_rag_system.chat_stream(
                user_input,
                st.session_state.conversation_history[:-1]  # 不包含当前消息
            ):
                # 检查是否为多跳检索通知
                if isinstance(chunk, dict) and chunk.get("type") == "multi_hop_notification":
                    # 显示多跳检索提示
                    if not multi_hop_notification_placeholder:
                        multi_hop_notification_placeholder = st.empty()
                    multi_hop_notification_placeholder.info("💭 **" + chunk.get("message", "您的问题相对复杂，正在执行深度思考推理…") + "**")
                    continue
                
                if is_final:
                    # 最终结果
                    result = chunk
                    full_response = result["answer"]
                    used_rag = result.get("used_rag", False)
                    sources = result.get("sources", [])
                    rag_reason = result.get("rag_reason", "")
                    is_meta_analysis = result.get("is_meta_analysis", False)
                    is_multi_hop = result.get("is_multi_hop", False)
                    
                    # 隐藏元文本分析提示（在显示回答之前清除提示）
                    if meta_notification_placeholder:
                        meta_notification_placeholder.empty()
                    
                    # 隐藏多跳检索提示（在显示回答之前清除提示）
                    if multi_hop_notification_placeholder:
                        multi_hop_notification_placeholder.empty()
                    
                    # 显示完整回答
                    message_placeholder.write(full_response)
                    
                    # 计算并记录总耗时
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    # 提取详细的时间统计
                    route_time = result.get("route_time", 0.0)
                    rewrite_time = result.get("rewrite_time", 0.0)
                    rag_time = result.get("rag_time", 0.0)
                    llm_time = result.get("llm_time", 0.0)
                    other_time = result.get("other_time", 0.0)
                    total_time = result.get("total_time", elapsed_time)
                    
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
                    
                    # 显示RAG信息
                    if used_rag:
                        with st.expander("📖 参考来源", expanded=True):
                            if sources:
                                st.info(f"**RAG判断理由:** {rag_reason}")
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"**来源 {i}:** {source.get('chapter', '未知章节')}")
                                    st.caption(source.get('content', '')[:200] + "...")
                            else:
                                st.warning("未找到相关文档")
                    else:
                        st.caption(f"💬 直接回答（未使用RAG） - {rag_reason}")
                else:
                    # 流式输出片段
                    # 在收到第一个流式chunk时，清除元文本分析提示和多跳检索提示
                    if not first_chunk_received:
                        if is_meta_analysis and meta_notification_placeholder:
                            meta_notification_placeholder.empty()
                        if multi_hop_notification_placeholder:
                            multi_hop_notification_placeholder.empty()
                        first_chunk_received = True
                    
                    full_response += chunk
                    message_placeholder.write(full_response + "▌")
            
            # 更新消息历史（确保保存当前回答）
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "used_rag": used_rag,
                "sources": sources,
                "is_meta_analysis": is_meta_analysis
            })
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
            
        except Exception as e:
            # 即使出错也记录耗时
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.error(f"[错误] 处理问题时出错: {str(e)}")
            logger.info(f"[完成时间] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
            logger.info(f"[总耗时] {elapsed_time:.2f} 秒")
            logger.info("=" * 80)
            
            error_msg = f"处理问题时出错: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "used_rag": False,
                "sources": [],
                "is_meta_analysis": False
            })

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "基于LLM+RAG的智能问答系统 | 支持《雪落成诗》和《影化成殇》"
    "</div>",
    unsafe_allow_html=True
)



