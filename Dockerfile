# 小说 RAG 系统 - 公网部署用（AI Builders / HF Spaces / Cloud Run）
# Python 3.11；千帆 Embedding；无本地大模型以控制镜像体积
# HF Spaces 要求以 UID 1000 运行，故创建非 root 用户
FROM python:3.11-slim

RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user:user requirements.txt .
USER user
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .
RUN mkdir -p runtime_data

# 平台注入 PORT；HF README app_port=8000；默认 8000
EXPOSE 8000

CMD sh -c "uvicorn web_chat:app --host 0.0.0.0 --port ${PORT:-8000}"
