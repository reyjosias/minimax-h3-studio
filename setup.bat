@echo off
title Configurar MiniMax H3 Local AI Studio
cd /d "%~dp0"
where py >nul 2>&1
if errorlevel 1 (
  python setup.py
) else (
  py -3 setup.py
)
if errorlevel 1 (
  echo.
  echo La configuracion no termino correctamente.
  pause
  exit /b 1
)
pause
