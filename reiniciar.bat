@echo off
title Reiniciar Local AI Studio
cd /d "%~dp0"
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8200" ^| findstr LISTENING') do taskkill /F /PID %%p >nul 2>&1
timeout /t 2 >nul
call start.bat
