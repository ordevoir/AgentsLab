@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0\generate_junctions.ps1"
echo.
echo Done. Press any key to close...
pause >nul
