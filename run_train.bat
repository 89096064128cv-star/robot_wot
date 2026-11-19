@echo off
chcp 65001 >nul
set PYTHONUTF8=1
title Обучение бота World of Tanks
cd /d "%~dp0"

python yolo_bot\train_yolo.py

pause