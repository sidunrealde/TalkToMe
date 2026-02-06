@echo off
:: TalkToMe - Quick Launch Script
:: Double-click this file to start the application

cd /d "%~dp0"

echo.
echo ========================================
echo    TalkToMe - AI Avatar Application
echo ========================================
echo.

:: Check if PowerShell execution policy allows scripts
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force; & '%~dp0start.ps1'"

pause
