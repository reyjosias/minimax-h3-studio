@echo off
title Reiniciar MiniMax H3 Studio
cd /d "%~dp0"
echo ============================================
echo   Reiniciando MiniMax H3 Studio...
echo ============================================
echo.

REM 1) Detener el servidor web del Studio si ya estaba corriendo (puerto 8199)
echo - Deteniendo la instancia anterior (puerto 8199)...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8199" ^| findstr LISTENING') do (
    taskkill /F /PID %%p >nul 2>&1
)
timeout /t 2 >nul

REM 2) Volver a arrancar: ComfyUI (si hace falta) + Studio + abrir el navegador
echo - Arrancando de nuevo...
echo.
"C:\Users\Rey\ComfyUI-Installs\Rey (1)\ComfyUI\.venv\Scripts\python.exe" launch.py
pause
