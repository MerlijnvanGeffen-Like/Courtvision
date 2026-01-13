@echo off
echo ========================================
echo    Courtvision - Starting Application
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Trying default Python path...
    set PYTHON_CMD=C:\Users\merli\AppData\Local\Programs\Python\Python311\python.exe
    if not exist "%PYTHON_CMD%" (
        echo ERROR: Python not found. Please install Python or update the path in run.bat
        pause
        exit /b 1
    )
) else (
    set PYTHON_CMD=python
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo [1/3] Starting API Server...
start "Courtvision API Server" cmd /k "%PYTHON_CMD%" api_server.py

REM Wait a bit for the API server to start
timeout /t 3 /nobreak >nul

echo [2/3] Installing webapp dependencies (if needed)...
cd webapp
if not exist node_modules (
    echo Installing npm packages...
    call npm install
)
cd ..

echo [3/3] Starting Webapp...
start "Courtvision Webapp" cmd /k "cd webapp && npm run dev"

echo.
echo ========================================
echo    Application Started!
echo ========================================
echo.
echo API Server: http://localhost:5000
echo Webapp: http://localhost:3000 (or check the webapp window)
echo.
echo Press any key to close this window (servers will keep running)...
pause >nul
