@echo off
title Local AI Studio
cd /d "%~dp0"
if not exist "studio_config.json" (
  echo Configuracion inicial necesaria...
  call setup.bat
  if errorlevel 1 exit /b 1
)
where py >nul 2>&1
if errorlevel 1 (
  python launch.py
) else (
  py -3 launch.py
)
pause
