@echo off
REM Quick start script for Road Safety Website (Windows)

echo.
echo ==========================================
echo Road Safety - Web Dashboard Setup
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

echo ✓ Python found:
python --version
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)

echo ✓ Dependencies installed successfully
echo.
echo ==========================================
echo ✓ Setup Complete!
echo ==========================================
echo.
echo To start the web application, run:
echo   python app.py
echo.
echo Then open your browser to:
echo   http://localhost:5000
echo.
echo For more information, see WEBSITE_README.md
echo.
pause
