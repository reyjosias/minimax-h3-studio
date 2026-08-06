@echo off
title Apagar MiniMax H3 Studio
cd /d "%~dp0"
echo ============================================
echo   Apagando MiniMax H3 Studio...
echo ============================================
echo.

REM 1) Liberar la GPU: cancelar cualquier generacion en ComfyUI (best-effort)
echo - Liberando la GPU (cancelando generaciones en curso)...
curl -s -X POST http://127.0.0.1:8188/interrupt >nul 2>&1
curl -s -X POST http://127.0.0.1:8188/queue -H "Content-Type: application/json" -d "{\"clear\":true}" >nul 2>&1

REM 2) Detener el servidor web del Studio (puerto 8199)
echo - Deteniendo el servidor del Studio (puerto 8199)...
set "FOUND="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8199" ^| findstr LISTENING') do (
    taskkill /F /PID %%p >nul 2>&1
    set "FOUND=1"
)
if defined FOUND (
    echo   Studio detenido correctamente.
) else (
    echo   El Studio no estaba corriendo.
)

echo.
echo Listo. ComfyUI sigue abierto en su propia ventana; cierrala si quieres apagarlo tambien.
timeout /t 3 >nul
