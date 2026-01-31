@echo off
chcp 65001 >nul
echo ========================================
echo 启动Web聊天界面
echo ========================================
echo.

REM 设置动态摘要环境变量（默认启用）
set ENABLE_DYNAMIC_SUMMARY=true
echo 动态摘要功能: 已启用（默认）
echo 提示: 如需禁用，请设置 ENABLE_DYNAMIC_SUMMARY=false
echo.

REM 启动Web聊天界面
echo 正在启动Web聊天界面...
python web_chat.py

pause
