# 小说 RAG 系统 - 公网部署用（AI Builders / HF Spaces / Cloud Run）
# Python 3.11；千帆 Embedding；无本地大模型以控制镜像体积
# HF Spaces 要求以 UID 1000 运行，故创建非 root 用户
FROM python:3.11-slim

RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_PREFER_BINARY=1

WORKDIR $HOME/app

COPY --chown=user:user requirements-deploy.lock .
USER user
# 非 root 装进用户目录。锁文件把传递依赖钉死，合计必须小于 256 MiB。
# 推送前运行 python scripts/check_deploy_budget.py。
RUN python -m pip install --no-cache-dir --user --upgrade pip \
    && python -m pip install --no-cache-dir --user --prefer-binary -r requirements-deploy.lock

COPY --chown=user:user . .
RUN mkdir -p runtime_data

# 平台注入 PORT；HF README app_port=8000；默认 8000
EXPOSE 8000

CMD sh -c "uvicorn web_chat:app --host 0.0.0.0 --port ${PORT:-8000}"
