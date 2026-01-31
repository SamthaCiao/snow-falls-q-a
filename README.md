# 小说RAG系统

基于 LangChain 与百度千帆 API 的小说检索问答系统，专门为《雪落成诗》和《影化成殇》设计。

## 功能特点

1. **智能分块**: 分别处理《雪落成诗》和《影化成殇》,不简单合并,确保人名不被切分在块边缘
2. **三重索引架构**: 语义向量层(ChromaDB) + 关键词索引层(BM25) + 摘要/主题层
3. **混合检索**: 结合语义向量搜索和BM25关键词搜索,提高检索准确性
4. **人物系统**: 集成轻量级人物画像系统,自动识别并检索相关人物信息,与RAG检索并行执行
5. **背景关联**: 分析《影化成殇》时自动结合《雪落成诗》的背景信息,但不输出未直接提及的内容
6. **行为分析模块**: 统一行为分析模块,支持行为解释类和情节推演类问题,调用人物画像库、时间线索引、元文本知识库、意象系统等数据源
7. **智能路由**: LLM自动判断问题类型,包括行为解释类、情节推演类、元文本分析类等多种类型
8. **多跳推理**: 集成ReasoningModule,支持复杂问题的多跳推理和二次检索
9. **结构化查询**: 使用StructuredQueryEngine精准提取JSON知识库字段,避免全文传递
10. **高效检索**: 使用向量数据库实现快速检索(目标<10秒)
11. **对话历史公开（可选）**: 支持将用户对话历史提交到服务端，所有用户、所有设备可见同一份历史列表，无需登录；首次进入网页时可选择「提交到公网」或「仅本地存储（localStorage）」。
12. **本地运行**: 一次性向量化后,后续检索完全本地运行
13. **API接口**: 提供FastAPI接口和Swagger UI,方便测试和集成

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥与数据源（本地开发）

- **密钥**：复制 `.env.example` 为 `.env` 或 `SUPER_MIND_API_KEY.env`，填入 `DEEPSEEK_API_KEY`、`BAIDU_API_KEY` 等（勿提交到 Git）。
- **数据源**：在项目根目录放置 `数据源/` 目录或 **`数据源.pack.zip`**（zip 二进制，可提交到仓库）；启动时若无 数据源/ 会从 数据源.pack.zip 解压使用。详见 [部署与密钥说明](docs/部署与密钥说明.md)。

**向量化**：本上线版本仅使用**百度千帆 API**（Qwen3-Embedding-8B）。在 `.env` 或 `SUPER_MIND_API_KEY.env` 中设置 `BAIDU_API_KEY`；模型 ID 可通过环境变量 `EMBEDDING_MODEL_BAIDU` 覆盖（默认 `qwen3-embedding-8b`）。API 文档: [千帆-向量](https://cloud.baidu.com/doc/qianfan-api/s/Fm7u3ropn)。

### 3. 启动Web聊天界面（推荐）

**Windows用户**:
```bash
run_chat.bat
```

**Linux/Mac用户**:
```bash
python web_chat.py
```

然后在浏览器中访问: `http://localhost:8000`

### 4. 或启动API服务（备用）

```bash
python api.py
```

服务将在 `http://localhost:8000` 启动，访问 `http://localhost:8000/docs` 查看API文档

## API接口说明（备用）

#### 回答问题
- **端点**: `POST /api/answer`
- **功能**: 根据问题返回答案
- **参数**:
  - `question`: 问题文本
  - `target_source`: 目标来源("雪落成诗"或"影化成殇"),可选
  - `use_background`: 是否使用背景信息,默认true
  - `top_k`: 返回top k个结果,默认5

#### 检索文档
- **端点**: `POST /api/search`
- **功能**: 检索相关文档片段
- **参数**: 同上

#### 重建索引
- **端点**: `POST /api/rebuild_index`
- **功能**: 重新构建向量索引(耗时较长)

## 编排与可观测性（LangChain 适配）

- **适配层**：`rag_modules/langchain_adapters.py` 将现有检索器封装为 LangChain `BaseRetriever`，支持链式编排与后续可观测（如 LangSmith）。
- **行为**：多路、多跳、行为/情节逻辑保持不变；仅将「按问题类型选检索器 → 检索 → 格式化」统一为通过 LangChain 检索接口调用；无 LangChain 时自动回退到直接调用原有检索器。
- **依赖**：`langchain-core`、`langchain-openai`（见 `requirements.txt`）。

## 技术架构

- **框架**: LangChain, FastAPI
- **Embedding**: 百度千帆 API（Qwen3-Embedding-8B），无本地 embedding
- **向量数据库**: ChromaDB (持久化存储)
- **关键词检索**: BM25 (rank-bm25) + Jieba分词
- **人物系统**: 基于JSON的人物画像知识库,自动分词和检索
- **行为分析模块**: 统一行为分析模块,支持行为解释类和情节推演类问题分析
- **知识库集成**: 集成人物画像库、时间线索引、元文本知识库、意象系统、心理学概念库、情节推演规则等
- **多跳推理**: ReasoningModule支持复杂问题的多跳推理判断和二次检索
- **结构化查询**: StructuredQueryEngine精准提取JSON知识库字段,避免全文传递,提高效率
- **分块策略**: 按章节分割,然后使用递归字符分割器,确保人名完整性
- **动态摘要**: 支持复杂问题的动态摘要生成(可选功能)
- **元文本分析**: 专门的元文本分析知识库和检索模块
- **智能路由**: LLM自动判断问题类型,包括行为解释类、情节推演类、元文本分析类等

## 设计说明

### 分块策略
- 两个文件分别处理,不合并
- 按章节标题分割,保持上下文完整性
- 使用重叠窗口保持语义连续性

### 检索逻辑
- **三重索引**: 语义向量、BM25关键词、摘要/主题三层索引协同工作
- **混合检索**: 结合语义相似度和关键词匹配,综合评分排序
- **人物识别**: 自动识别查询中的人物,并行检索人物画像和文档内容
- **智能判断**: LLM自动判断是否需要调用RAG
- **问题类型路由**: 自动判断问题类型,包括行为解释类、情节推演类、元文本分析类等
- **行为分析模块**: 对于行为解释类和情节推演类问题,调用统一行为分析模块,整合人物画像、时间线、元文本知识库、意象系统等数据源
- **知识库调用**: 行为解释类调用心理学概念库,情节推演类调用情节推演规则
- **多跳推理**: 对于复杂问题,ReasoningModule判断是否需要二次检索,生成链式查询意图
- **结构化查询**: StructuredQueryEngine根据查询词精准提取JSON知识库相关字段,避免全文传递
- **自动判断查询目标**: 根据关键词自动判断查询《雪落成诗》或《影化成殇》
- **背景关联**: 查询《影化成殇》时,自动检索《雪落成诗》中的相关背景
- **精确输出**: 只输出目标文档中直接提及的内容
- **来源标注**: 背景信息作为补充,明确标注来源

### 性能优化
- 一次性向量化,持久化存储；向量化仅使用千帆 API，无本地大模型/GPU 需求
- 使用ChromaDB实现快速相似度搜索
- 检索时间控制在10秒以内

## 公网部署（开放仓库）

- 仓库中**不包含** `.env`、明文 `数据源/` 目录（已写入 `.gitignore`）；可提交 **数据源.pack.zip**、**chroma_db/** 供部署使用，密钥通过 env_vars 注入。
- 部署前请阅读 [部署与密钥说明](docs/部署与密钥说明.md)，并按平台要求准备 Dockerfile、PORT、单进程等。
- 根目录已提供 `Dockerfile`、`.env.example`；部署时在平台设置 `env_vars`（如 `DEEPSEEK_API_KEY`、`BAIDU_API_KEY`、`DATA_SOURCE_DIR` 等）。

## 注意事项

1. **首次运行**: 会自动构建索引,可能需要几分钟时间(取决于硬件配置)
2. **索引存储**: 索引存储在 `./chroma_db` 目录（可通过环境变量 `CHROMA_DB_PATH` 覆盖）
3. **向量化**：仅使用百度千帆 API，需配置 `BAIDU_API_KEY`；无本地模型或 GPU 需求。
4. **重建索引**: 修改分块参数、切换embedding模型或修改人物系统后需要重建索引
5. **人物系统**: 人物画像数据存储在 `数据源/人物.json`（可通过环境变量 `DATA_SOURCE_DIR` 指定数据源目录）,系统会自动加载并集成到jieba分词词典
6. **动态摘要**: 默认启用，程序会自动判断复杂问题并使用动态摘要。如需禁用，设置环境变量 `ENABLE_DYNAMIC_SUMMARY=false`

## 项目结构

说明：明文 **数据源/** 目录不提交；可提交 **数据源.pack.zip**（zip 二进制）、**chroma_db/** 供部署使用。

```
.
├── 数据源.pack.zip                  # 数据源打包（zip 二进制，可提交；启动时自动解压使用）
├── chroma_db/                       # 向量与 BM25 缓存（可提交，供部署直接使用）
├── data_source_loader.py            # 数据源加载：无 数据源/ 时从 数据源.pack.zip 解压
├── rag_modules/                     # RAG检索模块
│   ├── base_retriever.py           # 基础检索器
│   ├── global_summary_retriever.py # 全局摘要检索器
│   ├── timeline_retriever.py       # 时间线检索器
│   ├── character_arc_retriever.py  # 人物关系检索器
│   ├── imagery_retriever.py        # 意象检索器
│   ├── meta_knowledge_retriever.py # 元文本知识检索器
│   ├── chapter_rag_retriever.py    # 章节RAG检索器
│   ├── chapter_summary_retriever.py # 章节摘要检索器
│   ├── dynamic_summary_generator.py # 动态摘要生成器
│   ├── behavior_analyzer.py        # 统一行为分析模块
│   ├── reasoning_module.py         # 多跳推理模块
│   └── structured_query_engine.py  # 结构化查询引擎
├── static/                         # 静态文件目录
│   └── chat.js                     # 前端JavaScript
├── templates/                      # HTML模板目录
│   └── chat.html                   # 聊天界面模板
├── rag_system.py                   # RAG系统核心模块
├── llm_rag_system.py              # LLM+RAG智能系统
├── web_chat.py                     # FastAPI Web聊天界面(主要)
├── chat_ui.py                      # Streamlit聊天界面(备用)
├── api.py                          # FastAPI API接口(备用)
├── rebuild_index.py                # 索引重建脚本
├── requirements.txt                # 依赖列表
├── README.md                       # 项目说明
├── 使用说明.md                     # 使用说明文档
└── 动态摘要测试问题.md             # 动态摘要测试问题列表
```

## 核心文件说明

- **rag_system.py**: RAG系统核心，负责文档加载、分块、向量化、检索
- **llm_rag_system.py**: LLM+RAG智能系统，负责路由判断、上下文检索、答案生成
- **web_chat.py**: FastAPI Web聊天界面，提供WebSocket流式输出和元文本分析支持
- **chat_ui.py**: Streamlit聊天界面，提供交互式聊天体验
- **api.py**: FastAPI API接口(备用)，提供RESTful API
- **rag_modules/reasoning_module.py**: 多跳推理模块，使用DeepSeek Reasoner进行推理判断和二次检索
- **rag_modules/structured_query_engine.py**: 结构化查询引擎，精准提取JSON知识库字段

## 示例请求

```python
import requests

# 回答问题
response = requests.post("http://localhost:8000/api/answer", json={
    "question": "您的问题",
    "use_background": True
})
print(response.json())
```
