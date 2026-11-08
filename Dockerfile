# 小说 RAG 系统 - 公网部署用（AI Builders / HF Spaces / Cloud Run）
# Python 3.11；千帆 Embedding；无本地大模型以控制镜像体积
# HF Spaces 要求以 UID 1000 运行，故创建非 root 用户
FROM python:3.11-slim

RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_PREFER_BINARY=1 \
    PIP_QUIET=1 \
    PIP_PROGRESS_BAR=off

WORKDIR $HOME/app
RUN mkdir -p runtime_data && chown -R user:user $HOME/app

COPY --chown=user:user requirements-deploy.lock .
USER user
# 非 root 装进用户目录。锁文件把传递依赖钉死，合计必须小于 256 MiB。
# 推送前运行 python scripts/check_deploy_budget.py。
# 平台构建日志只返回约 20KB。静默安装去掉 Collecting 行，并把 pip 的 stderr 并进 stdout，失败信息才不会被进度刷掉。
RUN python -m pip install --no-cache-dir --user --quiet --progress-bar off --upgrade pip 2>&1 \
    && python -m pip install --no-cache-dir --user --quiet --progress-bar off --prefer-binary -r requirements-deploy.lock 2>&1

COPY --chown=user:user . .

# 平台注入 PORT；HF README app_port=8000；默认 8000
EXPOSE 8000

CMD sh -c "uvicorn web_chat:app --host 0.0.0.0 --port ${PORT:-8000}"
