# 现有程序本地硬件依赖与 GitHub Pages 部署说明

## 一、除「数据源」外的本地/硬件依赖

**说明**：本上线版本**仅使用百度千帆 API 做向量化**，无本地 embedding（已移除 SentenceTransformer/GPU 相关逻辑）。

### 1. 磁盘（本地路径）

| 用途 | 路径/说明 |
|------|-----------|
| **ChromaDB 向量库** | `./chroma_db`（相对当前工作目录，持久化） |
| **BM25 缓存** | `./chroma_db/bm25_main_cache.pkl`、`bm25_sequel_cache.pkl` |
| **环境与密钥** | `SUPER_MIND_API_KEY.env`、`.env`（项目根或当前工作目录） |

所有路径均相对**进程当前工作目录**；数据源目录 `数据源/` 也相对该目录。

### 2. 内存

- Chroma 客户端 + 向量索引（部分在 Chroma 持久化文件中，查询时仍会占用内存）
- BM25：`bm25_main_texts`、`bm25_sequel_texts`、分词结果及 BM25Okapi 对象常驻内存
- FastAPI + LLMRAGSystem + 请求处理过程中的临时数据

### 3. 网络（运行时）

- **千帆 Embedding**：`https://qianfan.baidubce.com/v2/embeddings`（向量化仅千帆 API）
- **LLM**：DeepSeek/OpenAI 等（`OPENAI_API_BASE` + 对应 API Key）
- **可选**：百度千帆用于问题分类（若配置了 `BAIDU_API_KEY`）

### 4. 进程与运行时

- **Python 3.x** 常驻进程（uvicorn/FastAPI）；向量化仅调用千帆 API，无本地大模型或 GPU 需求。
- **流式响应**：通过线程 + 队列把同步 `chat_stream` 转成异步 SSE 输出

---

## 二、前端与后端的边界

- **前端**：`templates/chat.html`、`static/chat.js` 等；通过 **相对路径** 调用后端：
  - `POST /api/route-check`
  - `POST /api/chat`（流式 SSE）
- **后端**：`web_chat.py`（或 `api.py`）在**同一域名/同源**下提供上述接口；依赖本地磁盘（Chroma、BM25、数据源）、环境变量与千帆 Embedding API、远程 LLM。

因此：**数据源**只是依赖之一；**本地磁盘（Chroma、BM25、env）、内存** 为当前架构下的本地/硬件依赖；**向量化仅使用千帆 API**，无本地 GPU/CPU 需求。

---

## 三、GitHub Pages 的约束与所需架构改动

### 3.1 GitHub Pages 能做什么

- 只提供**静态资源**：HTML、CSS、JS、图片等。
- **不能**：跑 Python、FastAPI、读写服务器磁盘、读环境变量、维护 WebSocket/长连接服务。

因此：**当前后端（RAG + LLM + Chroma + BM25）不能部署在 GitHub Pages 上**，只能部署在**其它支持运行 Python 与持久化存储的服务**上。

### 3.2 推荐架构：前后端分离 + 后端单独部署

```
[ GitHub Pages ]                     [ 后端服务器 / 云服务 ]
  静态页面                              Python + FastAPI
  (HTML/CSS/JS)                        ChromaDB + BM25 + 数据源
        |                                       |
        |   HTTPS 请求                           | 读磁盘、调千帆/LLM
        +---------------- 调用 -----------------+
                    API Base URL
              (如 https://your-api.example.com)
```

需要做的**架构改动**如下。

---

### 3.3 必须做的改动

#### 1）前端：API 基地址可配置（已实现）

- **现状**：`chat.js` 已通过 `getApiBase()` 使用可配置基地址；同源时 `window.__API_BASE__` 为空即可。
- **用法**：部署到 GitHub Pages 时，在 `templates/chat.html` 中（或部署后的 HTML 里）设置：
  ```html
  <script>window.__API_BASE__ = 'https://your-backend.example.com';</script>
  ```
  所有 `/api/route-check`、`/api/chat` 请求会发往该基地址。

#### 2）后端：CORS 放行 GitHub Pages 域名（已实现）

- **现状**：`web_chat.py` 已根据环境变量 `CORS_ORIGINS` 添加 CORS 中间件；未设置时不影响同源本地使用。
- **用法**：部署后端时设置环境变量，例如：
  ```bash
  CORS_ORIGINS=https://username.github.io,https://your-custom-domain.com
  ```
  多个来源用逗号分隔。

#### 3）后端部署到「非 GitHub Pages」的服务器

- 后端需：**Python 运行环境 + 持久化磁盘（或可挂载卷）**。
- 把**数据源目录**、**Chroma 目录**（及 BM25 缓存）一起部署或挂载到同一台机器（或同一存储），保证工作目录/路径一致（或通过配置统一为绝对路径）。
- 在部署环境中配置环境变量（或 `.env`）：`DEEPSEEK_API_KEY`、`BAIDU_API_KEY` 等；**不要**把密钥提交到 GitHub。

可选部署方式示例：

- **VPS / 云主机**：直接跑 `uvicorn web_chat:app`，用 Nginx 反代并配 HTTPS。
- **Railway / Render / Fly.io**：支持持久化卷的，把 `chroma_db`、`数据源` 放在卷上；启动命令为当前项目的 `uvicorn`。
- **Serverless（如 Vercel Functions）**：无持久化磁盘，需把 **Chroma 换成远程向量库**（如 Pinecone、Weaviate、Chroma Cloud），并考虑冷启动与超时时间是否满足 RAG+LLM 延迟需求。

---

### 3.4 可选但建议的改动

#### 1）路径与配置统一

- 将 `./chroma_db`、`./数据源`、BM25 缓存目录等改为**可由环境变量覆盖**（如 `CHROMA_DB_PATH`、`DATA_SOURCE_DIR`），便于在不同环境（本机、Docker、云）使用同一套代码。

#### 2）若后端用 Serverless 或无盘实例

- 用**远程向量库**替代本地 Chroma（需改 `rag_system.py` 中 Chroma 的初始化和调用）。
- 数据源和预计算索引可在**构建/发布阶段**上传到该向量库，运行时只做查询与 LLM 调用。

#### 3）前端构建与发布

- 若希望用 TypeScript/打包工具：可单独建前端工程，构建产物部署到 GitHub Pages；`API_BASE` 在构建时注入（如 Vite 的 `import.meta.env.VITE_API_BASE`）。
- 若保持单 HTML+JS：只需在 HTML 里用一行脚本设置 `window.__API_BASE__`，并在 `chat.js` 里把所有 `/api/...` 改为 `(window.__API_BASE__ || '') + '/api/...'`。

---

## 四、小结

| 类别 | 内容 |
|------|------|
| **除数据源外的本地依赖** | 磁盘（Chroma、BM25 缓存、env 文件）、内存、网络（千帆 Embedding API + LLM）；无本地 GPU/embedding 需求 |
| **GitHub Pages 能部署的** | 仅前端静态页面（HTML/CSS/JS） |
| **为用 GitHub Pages 必须做的** | ① 前端改为请求「可配置的 API 基地址」 ② 后端配置 CORS ③ 后端部署到支持 Python + 磁盘（或远程向量库）的服务器，并配置密钥与数据源/Chroma 路径 |

按上述拆分后，**页面**可放在 GitHub Pages，**RAG+LLM 逻辑**留在独立后端，即可在满足 GitHub Pages 限制的前提下完成部署。
