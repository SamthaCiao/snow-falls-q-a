@echo off
chcp 65001 >nul
set REPO_URL=https://github.com/SamthaCiao/snow-falls-q-a.git
cd /d "%~dp0"

if not exist ".git" (
    echo [1/5] 初始化 Git 仓库...
    git init
) else (
    echo [1/5] 已是 Git 仓库，跳过 init
)

echo [2/5] 添加所有修改...
git add -A

echo [3/5] 提交（共享历史功能）...
git commit -m "feat: 共享历史对话 - 所有用户/设备可见，无需登录，服务端内存存储" 2>nul
if errorlevel 1 (
    git status -sb
    echo.
    echo 没有新修改需要提交，或已是最新。若要强制提交空提交可运行: git commit --allow-empty -m "chore: sync"
    goto remote
)
echo 提交完成。

:remote
echo [4/5] 配置远程 origin...
git remote remove origin 2>nul
git remote add origin %REPO_URL%
echo 远程地址: %REPO_URL%

echo [5/5] 推送到 GitHub...
git branch -M main 2>nul
git push -u origin main 2>nul
if errorlevel 1 (
    echo.
    echo 若远程已有历史且与本地不同，可先拉取再推送:
    echo   git pull origin main --rebase
    echo   git push -u origin main
    echo.
    echo 或允许合并不相关历史（会保留两边提交）:
    echo   git pull origin main --allow-unrelated-histories
    echo   git push -u origin main
)
echo.
echo 完成。仓库: https://github.com/SamthaCiao/snow-falls-q-a
pause
