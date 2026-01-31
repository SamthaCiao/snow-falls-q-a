# env_vars 与数据源配置说明（傻瓜式）

根据 [AI Builder 平台 OpenAPI](https://www.ai-builders.com/resources/students-backend/openapi.json)，**env_vars、DATA_SOURCE_DIR、CHROMA_DB_PATH 都没有单独的“配置页面”**，都是在**同一次“部署”操作**里一起配置的。下面按“在哪儿配、怎么配”一步步说明。

---

## 一、env_vars 在哪儿配置？

**答：在「发起部署」时配置。**

- 平台通过 **POST /v1/deployments** 接口部署你的服务。
- 请求体里除了 `repo_url`、`service_name`、`branch`，还可以带一个 **`env_vars`** 对象。
- 你填在 `env_vars` 里的键值对，会变成容器里的**环境变量**（例如应用里用 `os.getenv("DEEPSEEK_API_KEY")` 读到的就是这里填的值）。

也就是说：  
**没有**单独的“环境变量设置页”或“密钥管理页”，**只有**在“发部署请求”的那一步，把 env_vars 写在请求体里。

---

## 二、具体怎么配？（傻瓜式步骤）

### 直接手写请求体或由平台配置

- 你或 AI 在调用 **POST /v1/deployments** 时，直接写一个 JSON 请求体，例如：
  ```json
  {
    "repo_url": "https://github.com/SamthaCiao/snow-falls-q-a",
    "service_name": "snow-falls-q-a",
    "branch": "main",
    "env_vars": {
      "DEEPSEEK_API_KEY": "你的DeepSeek密钥",
      "BAIDU_API_KEY": "你的千帆密钥"
    }
  }
  ```
- 密钥不要写进代码仓库，可以只在本地使用，或由部署平台在创建服务时配置 env_vars。

---

## 三、DATA_SOURCE_DIR、CHROMA_DB_PATH 在哪儿配置？

**答：和上面一样，都在同一次部署请求的 `env_vars` 里。**

- **DATA_SOURCE_DIR**：应用用来找「数据源」目录的环境变量；若部署平台**支持挂载目录/卷**，你把数据源挂载到容器里某路径（如 `/app/数据源`），则在 **env_vars** 里加一项：`"DATA_SOURCE_DIR": "/app/数据源"`。
- **CHROMA_DB_PATH**：应用用来找 chroma 索引目录的环境变量；若平台支持挂载且你把 chroma_db 挂载到容器里某路径（如 `/app/chroma_db`），则在 **env_vars** 里加：`"CHROMA_DB_PATH": "/app/chroma_db"`。

注意：

- 当前部署方式（从 GitHub 拉代码、用 Dockerfile 构建）**不会**自动带上你本地的 `数据源/` 和 `chroma_db/`（它们已在 .gitignore 里）。
- 若平台**不支持**挂载卷或没有提供“数据源/索引”的注入方式，容器里就没有这些目录，应用启动后可能报错“找不到数据源”。  
- **若你不确定平台是否支持挂载、或不知道如何挂载**：请**联系老师/平台文档**确认「如何提供数据源与 chroma_db」（例如是否支持卷挂载、是否有示例数据等）。本说明只告诉你：**若支持挂载，则 DATA_SOURCE_DIR / CHROMA_DB_PATH 就在同一次部署请求的 env_vars 里配置即可。**

---

## 四、小结（傻瓜式记忆）

| 你想配置的       | 在哪里配置                         |
|------------------|------------------------------------|
| **env_vars**     | 发起部署时，请求体里的 **env_vars**（或部署平台提供的环境变量配置）。 |
| **DATA_SOURCE_DIR** | 同上，写在 **env_vars** 里；且需平台支持挂载数据源目录。     |
| **CHROMA_DB_PATH**  | 同上，写在 **env_vars** 里；且需平台支持挂载 chroma_db。     |

没有单独的“环境变量页面”或“密钥管理页面”，**都是在「发部署请求」或「在平台配置服务」时，通过请求体里的 env_vars 或平台的环境变量配置一起传进去的**。

---

## 五、老师/平台是否支持挂载数据源？（结论）

根据 **OpenAPI**（[openapi.json](https://www.ai-builders.com/resources/students-backend/openapi.json)）：

- 部署接口 **POST /v1/deployments** 的请求体只有：`repo_url`、`service_name`、`branch`、`port`、`env_vars`、`streaming_log_timeout_seconds`。
- **没有任何“挂载目录 / 卷 / 文件注入”类参数**。流程是：平台 clone 你给的公开仓库 → 按 Dockerfile 构建镜像 → 在 Koyeb 上跑容器，只把 `env_vars` 注入为环境变量。

**结论：当前文档和接口下，老师/平台不提供“挂载数据源目录”的能力。** 若后续老师或平台增加了挂载方式，再在 env_vars 里配 `DATA_SOURCE_DIR` / `CHROMA_DB_PATH` 即可。

---

## 六、GitHub Secret 方式是否可行？

**在当前部署流程下，用 GitHub Secrets 把数据源“传”给平台，不可行。**

原因简要说明：

- 部署是由 **你（或脚本）调用 POST /v1/deployments** 触发的；平台收到请求后去 **clone 你指定的公开仓库**，再构建、运行。
- **GitHub Secrets** 只在 **GitHub Actions** 里可用，部署时平台**不会**执行你的 Actions，也**拿不到**你的 Secrets。
- 所以无法通过“在仓库里用 Secret 存数据源、让平台拉取”的方式把数据源交给容器。

若你希望**不把数据源放进公开仓库**，又要在公网跑起来，可行方向只有：

1. **联系老师**：问是否提供挂载/卷、或是否有示例数据源/私有存储可用。
2. **自建“启动时下载数据源”**：把数据源放在私有地址（如带 token 的 URL 或对象存储），在 **env_vars** 里传下载地址和鉴权信息，在应用或镜像**启动脚本**里用这些环境变量下载并解压到 `数据源/`。这需要改 Dockerfile 或入口脚本，属于代码改动。

---

## 七、chroma_db：是否需要传、能否公网传？

- **运行时是否需要 chroma_db？**  
  需要。当前 RAG 检索、BM25 等依赖 chroma_db 里的向量与缓存；没有 chroma_db 就无法正常做检索和问答。

- **能否不传 chroma_db？**  
  只有在你改成“不依赖本地 chroma_db”的部署方式（例如改用远程向量库）时才可以不传；当前架构下**需要**传。

- **若不需要 rebuild_index、索引不变，能否在公网传？**  
  可以。做法是：**在本地构建好 chroma_db 后，把它纳入仓库并提交到 GitHub**（从 `.gitignore` 里去掉 `chroma_db/`，再 `git add chroma_db`、commit、push）。这样平台 clone 时就会带上 chroma_db，容器里就有索引，无需挂载。  
  注意：仓库会变大，且 chroma_db 内容会公开；若你能接受“索引公开、且不常重建”，这种“chroma_db 随仓库公网传”是可行且简单的。

- **数据源 能否也这样公网传？**  
  应用运行时同样会读 **数据源/**（人物.json、元文本知识库、时间线等）。若你把 **数据源/** 也提交到公开仓库，从技术上讲可以跑起来，但 **数据源** 通常涉及版权/隐私，**不建议**把整份数据源公开在 GitHub 上。更稳妥的是：只把 chroma_db 纳入仓库（或按上面“启动时下载”的方式提供数据源），数据源继续用私有方式或问老师怎么提供。
