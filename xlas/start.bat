@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py %*
) else (
    "..\env\Scripts\python.exe" app.py %*
)
if errorlevel 1 (
    echo.
    echo Khong khoi dong duoc XLAS. Xem huong dan tao moi truong trong README.md.
    pause
)
