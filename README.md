# 雪落回声

基于 LangChain 与 DeepSeek / 千帆 API 的小说检索问答，面向《雪落成诗》与《影化成殇》。Web 端产品名 **雪落回声**；核心是 RAG 问答，并带有异步 **阅读轨迹** 与 **探索星图**（见 [探索星图说明](docs/探索星图说明.md)）。

## 功能概览

- **混合检索**：Chroma 向量 + BM25 + 章节/摘要/时间线等多路检索
- **人物与行为分析**：人物画像、意象、心理学概念、情节推演规则等结构化库
- **智能路由**：按问题类型选检索器；复杂问题可走多跳推理
- **阅读轨迹（用户观测）**：独立进程异步归纳「聊过什么、关注什么」，SQLite 存储，可选 Google 登录
- **探索星图**：轨迹映射到 ∞ 形星云路径，进度揭示亮区；悬停人物看关联事件
- **引文校验**：回答中带引号的句子会对照 RAG 原文语料校验，失败则打回重写一次（`ENABLE_QUOTE_VALIDATION`）
- **本地向量索引**：首次向量化后检索以本地 Chroma 为主（embedding 走千帆 API）

更细的操作说明见 [使用说明.md](使用说明.md)；观测与存储见 [用户观测与存储说明](docs/用户观测与存储说明.md)。

## 近期能力（2026 上）

- 用户观测模块：主进程写事件、observer 进程抽取 state，`retrieval_hints` 仅作检索扩展，不往回答里贴标签
- 探索星图：星云主体底图 + 静态叙事星表 + 按对话点亮；与轨迹共用同一套 session 数据
- 英文/混合问句：查询改写与多 query 并行检索（2026-04 起，细节见 git 历史）

## 安装与启动

```bash
pip install -r requirements.txt
```

### 配置

- 复制 `.env.example` 为 `.env` 或 `SUPER_MIND_API_KEY.env`，填入 `DEEPSEEK_API_KEY`、`BAIDU_API_KEY` 等（勿提交 Git）。
- 数据源：根目录放 `数据源/` 或 **`数据源.pack.zip`**；无目录时启动自动解压。
- 向量化仅用千帆 API（默认 `qwen3-embedding-8b`），见 [千帆向量文档](https://cloud.baidu.com/doc/qianfan-api/s/Fm7u3ropn)。

### 日常开发（两个终端）

```bash
# 终端 1
python web_chat.py          # 或 run_chat.bat

# 终端 2
python user_observer.py     # 或 run_observer.bat
```

浏览器：`http://localhost:8000`

探索星图、轨迹抽取、API 字段见 [用户观测与存储说明](docs/用户观测与存储说明.md) 与 [探索星图说明](docs/探索星图说明.md)。

### 备用入口

```bash
python chat_ui.py           # Streamlit 界面（无观测/星图）
```

API 文档：启动 `web_chat.py` 后访问 `http://localhost:8000/docs`。

## API 说明

**日常开发以 `web_chat.py` 为准**（`http://localhost:8000/docs`）。主要端点：

| 端点 | 说明 |
|------|------|
| `POST /api/chat` | SSE 流式问答（含 RAG 路由、多跳、引文校验） |
| `POST /api/route-check` | 快速路由预判（元文本等 UI 提示） |
| `GET/POST /api/shared_history` | 内存共享对话列表（重启清空） |
| `GET /api/auth/me` | 当前用户（Google 或匿名） |
| `GET /auth/login` · `/auth/callback` | Google OAuth（可选） |
| `POST /api/observation/session` | 创建观测 session |
| `GET /api/observation/{id}` | 阅读轨迹 state |
| `GET /api/observation/{id}/starmap` | 探索星图 payload |
| `GET /api/observation/catalog` | 静态星表 + 背景图 URL |
| `POST /api/observation/trigger` | 手动重抽取轨迹 |
| `POST /api/observation/correct` | 用户修正可见轨迹文案 |
| `WebSocket /ws` | 流式（与 `/api/chat` 同能力，含观测写入） |

### 回答校验

- **引文**：带引号句子对照 RAG 原文语料；失败最多打回重写 2 次，仍失败则剥离引号
- **无引号事实句**：含具体情节/时间/行为的陈述需与检索 context 有关键词重叠；失败打回重写，仍失败则追加核实提示
- 可通过 `ENABLE_QUOTE_VALIDATION=false` 关闭（不推荐）

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
- **分块策略**: 按章节分割,然后使用递归字符分割器,确保人名完整性（Web 主路径默认 chunk_size=400）
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

- 勿提交 `.env`、`runtime_data/`、明文 `数据源/`；可提交 `数据源.pack.zip`、`chroma_db/`。
- 根目录有 `Dockerfile`、`.env.example`；密钥走平台环境变量（`DEEPSEEK_API_KEY`、`BAIDU_API_KEY` 等）。
- **Google OAuth 公网**：设置 `APP_PUBLIC_URL=https://你的后端域名`（与 MCP/平台注入的公网 URL 一致），并在 Google Console 添加 `${APP_PUBLIC_URL}/auth/callback`。

## 注意事项

1. **首次运行**: 会自动构建索引,可能需要几分钟时间(取决于硬件配置)
2. **索引存储**: 索引存储在 `./chroma_db` 目录（可通过环境变量 `CHROMA_DB_PATH` 覆盖）
3. **向量化**：仅使用百度千帆 API，需配置 `BAIDU_API_KEY`；无本地模型或 GPU 需求。
4. **重建索引**: 修改分块参数、切换embedding模型或修改人物系统后需要重建索引
5. **人物系统**: 人物画像数据存储在 `数据源/人物.json`（可通过环境变量 `DATA_SOURCE_DIR` 指定数据源目录）,系统会自动加载并集成到jieba分词词典
6. **动态摘要**: 默认启用，程序会自动判断复杂问题并使用动态摘要。如需禁用，设置环境变量 `ENABLE_DYNAMIC_SUMMARY=false`

## 项目结构

说明：明文 **数据源/** 不提交；可提交 **数据源.pack.zip**、**chroma_db/** 供部署。

```
.
├── web_chat.py / user_observer.py    # 主对话 + 观测（各开一个终端）
├── auth/                             # Google OAuth（可选）
├── config/observer_config.yaml       # 观测轮询/阈值
├── rag_system.py / llm_rag_system.py
├── rag_modules/                      # 检索、行为分析、推理、引文校验
├── observation/                      # 观测 DB、抽取、星图 render
│   ├── star_map.py, star_map_render.py, nebula_starmap.py
│   └── db.py, extractor.py, observer.py, …
├── static/
│   ├── chat.js, star_map_gl.js
│   ├── star_map_demo.html
│   ├── data/exploration_star_catalog.json
│   └── images/exploration_nebula_bg.webp
├── scripts/
│   ├── resnap_starmap_paths.py
│   ├── setup_nebula_starmap.py
│   └── build_starmap_demo.py
├── templates/chat.html
├── docs/
│   ├── 用户观测与存储说明.md
│   └── 探索星图说明.md
├── runtime_data/observation.db       # 本地生成，不提交
├── chroma_db/
└── 数据源.pack.zip
```

## 核心文件说明

- **web_chat.py**：Web 聊天、观测事件写入、星图 API
- **user_observer.py**：异步抽取阅读轨迹
- **llm_rag_system.py**：路由、检索编排、生成回答
- **rag_system.py**：分块、向量化、Chroma 索引
- **observation/**：SQLite、state 抽取、`retrieval_hints`、星图 render
- **static/star_map_gl.js**：探索星图前端

详细 API 与字段见 [用户观测与存储说明](docs/用户观测与存储说明.md)。

## 示例请求

```python
import json
import requests

# 主入口：SSE 流式问答（web_chat.py）
resp = requests.post(
    "http://localhost:8000/api/chat",
    json={"message": "夏空山对雪遗诗的态度如何变化？", "session_id": None},
    headers={"X-Anonymous-User-Id": "your-stable-anon-id"},  # OAuth 启用时建议固定匿名 ID
    stream=True,
)
for line in resp.iter_lines():
    if line and line.startswith(b"data: "):
        print(json.loads(line[6:]))
```
