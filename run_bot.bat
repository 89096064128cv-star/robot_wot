@echo off
chcp 65001 >nul
set PYTHONUTF8=1
title Бот World of Tanks
cd /d "%~dp0"

echo ========================================
echo   Запуск бота: ангар, очередь и YOLO-бой...
echo   Остановка: закройте окно или Ctrl+C
echo ========================================
echo.

python -u inference.py

echo.
echo ========================================
echo   Бот остановлен. Код выхода: %errorlevel%
echo ========================================
pause