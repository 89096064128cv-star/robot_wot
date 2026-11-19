@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
python -u wot_bot_v2\route_recorder.py %*
if errorlevel 1 (
    echo.
    echo Не удалось запустить запись маршрута.
    pause
)
