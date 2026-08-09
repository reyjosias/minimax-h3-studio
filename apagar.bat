@echo off
title Apagar Local AI Studio
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8200" ^| findstr LISTENING') do taskkill /F /PID %%p >nul 2>&1
echo Studio detenido. ComfyUI permanece abierto.
timeout /t 2 >nul
