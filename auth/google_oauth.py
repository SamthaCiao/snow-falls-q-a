"""Google OAuth 松耦合身份验证（可选启用）。"""
from __future__ import annotations

import logging
import os
import secrets
import uuid
from typing import Any, Dict, Optional

from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import HTTPException, Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from starlette.responses import RedirectResponse

logger = logging.getLogger(__name__)

OAUTH_COOKIE_NAME = "novel_rag_session"
ANONYMOUS_HEADER = "X-Anonymous-User-Id"
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 天

# 必须在读取 GOOGLE_* 之前加载 env（import 时 web_chat 尚未 load_dotenv）
load_dotenv("SUPER_MIND_API_KEY.env")
load_dotenv(".env")

oauth = OAuth()
_oauth_registered = False


def _serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET") or os.environ.get("DEEPSEEK_API_KEY") or "dev-insecure-change-me"
    return URLSafeTimedSerializer(secret, salt="novel-rag-auth")


def is_google_auth_enabled() -> bool:
    return bool(os.environ.get("GOOGLE_CLIENT_ID") and os.environ.get("GOOGLE_CLIENT_SECRET"))


def ensure_oauth_registered() -> None:
    """延迟注册 OAuth 客户端（env 加载后再 register）。"""
    global _oauth_registered
    if _oauth_registered:
        return
    if not is_google_auth_enabled():
        return
    oauth.register(
        name="google",
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    _oauth_registered = True
    logger.info("Google OAuth 客户端已注册")


def create_session_token(payload: Dict[str, Any]) -> str:
    return _serializer().dumps(payload)


def read_session_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        return _serializer().loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


def resolve_user_id(request: Request) -> str:
    """
    解析当前用户 ID：
    - 已登录：Google sub
    - 未启用 OAuth：匿名 ID（header 或 cookie 中的 anon id）
    """
    return resolve_user_id_from_headers(
        headers=request.headers,
        cookies=request.cookies,
    )


def resolve_user_id_from_headers(*, headers, cookies) -> str:
    """从 headers/cookies 解析用户 ID（Request / WebSocket 共用）。"""
    cookie = cookies.get(OAUTH_COOKIE_NAME) if cookies else None
    if cookie:
        data = read_session_token(cookie)
        if data and data.get("user_id"):
            return str(data["user_id"])

    if is_google_auth_enabled():
        anon = headers.get(ANONYMOUS_HEADER) or (cookies.get("novel_rag_anon_id") if cookies else None)
        if anon:
            return f"anon_{anon}"
        return f"anon_{uuid.uuid4().hex[:16]}"

    anon = headers.get(ANONYMOUS_HEADER) or (cookies.get("novel_rag_anon_id") if cookies else None)
    if anon:
        return f"anon_{anon}"
    return "anon_local"


def get_oauth_redirect_uri(request: Optional[Request] = None) -> str:
    """
    OAuth 回调 URI：优先 GOOGLE_REDIRECT_URI，其次 APP_PUBLIC_URL/auth/callback，
    最后使用当前请求的 url_for（本地开发）。
    """
    explicit = os.environ.get("GOOGLE_REDIRECT_URI", "").strip()
    if explicit:
        return explicit
    public_base = os.environ.get("APP_PUBLIC_URL", "").strip().rstrip("/")
    if public_base:
        return f"{public_base}/auth/callback"
    if request is not None:
        return str(request.url_for("auth_callback"))
    return "http://localhost:8000/auth/callback"


def get_user_profile(request: Request) -> Dict[str, Any]:
    cookie = request.cookies.get(OAUTH_COOKIE_NAME)
    if cookie:
        data = read_session_token(cookie)
        if data:
            return {
                "user_id": data.get("user_id"),
                "email": data.get("email"),
                "display_name": data.get("display_name"),
                "auth_provider": data.get("auth_provider", "google"),
                "authenticated": data.get("auth_provider") == "google",
            }
    uid = resolve_user_id(request)
    return {
        "user_id": uid,
        "email": None,
        "display_name": None,
        "auth_provider": "anonymous",
        "authenticated": False,
    }


async def google_login(request: Request):
    if not is_google_auth_enabled():
        raise HTTPException(status_code=501, detail="Google OAuth 未配置")
    ensure_oauth_registered()
    if not _oauth_registered:
        raise HTTPException(status_code=501, detail="Google OAuth 客户端注册失败")
    redirect_uri = get_oauth_redirect_uri(request)
    return await oauth.google.authorize_redirect(request, redirect_uri)


async def google_callback(request: Request):
    if not is_google_auth_enabled():
        raise HTTPException(status_code=501, detail="Google OAuth 未配置")
    ensure_oauth_registered()
    if not _oauth_registered:
        raise HTTPException(status_code=501, detail="Google OAuth 客户端注册失败")
    token = await oauth.google.authorize_access_token(request)
    userinfo = token.get("userinfo")
    if not userinfo:
        raise HTTPException(status_code=400, detail="无法获取 Google 用户信息")

    from observation.db import get_database

    user_id = str(userinfo["sub"])
    email = userinfo.get("email")
    name = userinfo.get("name")
    get_database().upsert_user(user_id, email=email, display_name=name, auth_provider="google")

    session_payload = {
        "user_id": user_id,
        "email": email,
        "display_name": name,
        "auth_provider": "google",
        "nonce": secrets.token_hex(8),
    }
    token_str = create_session_token(session_payload)
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        OAUTH_COOKIE_NAME,
        token_str,
        httponly=True,
        samesite="lax",
        max_age=SESSION_MAX_AGE,
        secure=os.environ.get("AUTH_COOKIE_SECURE", "").lower() in ("1", "true", "yes"),
    )
    return response


def logout_response() -> RedirectResponse:
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(OAUTH_COOKIE_NAME)
    return response
