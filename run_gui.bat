@echo off
chcp 65001 >nul
set PYTHONUTF8=1
title Управление ботом World of Tanks
cd /d "%~dp0"

python -u bot_gui.py

if errorlevel 1 (
    echo.
    echo Не удалось запустить графический интерфейс.
    pause
)
