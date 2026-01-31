# 小说 RAG 系统 - 公网部署用
# 使用 Python 3.11，默认依赖千帆 Embedding，不包含本地大模型以控制镜像体积
FROM python:3.11-slim

WORKDIR /app

# 安装依赖（不复制 数据源/chroma_db，由运行时挂载或注入）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码（.dockerignore 建议排除 数据源、chroma_db、.env、test）
COPY . .

# 平台会设置 PORT，默认 8000
EXPOSE 8000

# 使用 shell 形式以正确展开 PORT 环境变量
CMD sh -c "uvicorn web_chat:app --host 0.0.0.0 --port ${PORT:-8000}"
