@echo off
chcp 65001 >nul
echo ========================================
echo 启动用户观测模块（独立进程）
echo ========================================
echo.
echo 数据目录: runtime_data\ （SQLite，已在 .gitignore）
echo 配置: config\observer_config.yaml
echo.
python user_observer.py
pause
