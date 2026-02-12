@echo off
setlocal enabledelayedexpansion

cls
echo ================================================================
echo   MOTO SYSTEM LAUNCHER
echo ================================================================
echo.

REM ================================================================
REM STEP 1: Check Python Installation
REM ================================================================
echo [1/8] Checking Python installation...
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Python is not installed or not in PATH
    echo ============================================================
    echo.
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    echo IMPORTANT: Check 'Add Python to PATH' during installation
    echo.
    pause
    exit /b 1
)
python --version
echo Python found!
echo.

REM ================================================================
REM STEP 2: Check Node.js Installation
REM ================================================================
echo [2/8] Checking Node.js installation...
where node >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Node.js is not installed or not in PATH
    echo ============================================================
    echo.
    echo Please install Node.js 16+ from: https://nodejs.org/
    echo.
    pause
    exit /b 1
)
call node --version
if errorlevel 1 (
    echo ERROR: node --version failed
    pause
    exit /b 1
)
call npm --version
if errorlevel 1 (
    echo ERROR: npm --version failed
    pause
    exit /b 1
)
echo Node.js found!
echo.

REM ================================================================
REM STEP 3: Create Necessary Directories & Clean ChromaDB
REM ================================================================
echo [3/8] Creating necessary directories...
if not exist "backend\data" mkdir "backend\data"
if not exist "backend\data\user_uploads" mkdir "backend\data\user_uploads"
if not exist "backend\logs" mkdir "backend\logs"

REM Clean ChromaDB on startup to prevent corruption issues
if exist "backend\data\chroma_db" (
    echo Cleaning ChromaDB database...
    rmdir /s /q "backend\data\chroma_db"
    echo ChromaDB cleaned!
)

echo Directories created successfully!
echo.

REM ================================================================
REM STEP 4: Install Python Dependencies
REM ================================================================
echo [4/8] Installing Python dependencies...
echo This may take a few minutes if this is your first time...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Failed to install Python dependencies
    echo ============================================================
    echo.
    echo Please check:
    echo - Internet connection is working
    echo - You have permission to install packages
    echo - requirements.txt exists in the current directory
    echo.
    pause
    exit /b 1
)
echo Python dependencies installed successfully!
echo.

REM ================================================================
REM STEP 5: Install Node.js Dependencies
REM ================================================================
echo [5/8] Installing Node.js dependencies...
echo This may take a few minutes if this is your first time...
echo.
if not exist "frontend" (
    echo.
    echo ============================================================
    echo ERROR: frontend directory not found!
    echo ============================================================
    echo.
    echo Make sure you're running this from the project root directory.
    echo.
    pause
    exit /b 1
)
pushd frontend
call npm install
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Failed to install Node.js dependencies
    echo ============================================================
    echo.
    echo Please check:
    echo - Internet connection is working
    echo - package.json exists in frontend directory
    echo.
    popd
    pause
    exit /b 1
)
popd
echo Node.js dependencies installed successfully!
echo.

REM ================================================================
REM STEP 6: Check LM Studio (optional - OpenRouter is an alternative)
REM ================================================================
echo [6/8] Checking LM Studio...
echo.

REM Check if LM Studio is responding
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://127.0.0.1:1234/v1/models' -TimeoutSec 3 -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo ================================================================
    echo NOTE: LM Studio is not detected on http://127.0.0.1:1234
    echo ================================================================
    echo.
    echo This is OK! You have two options for AI models:
    echo.
    echo   Option 1: LM Studio (Local)
    echo     - Download from: https://lmstudio.ai/
    echo     - Load a model and start the Local Server
    echo.
    echo   Option 2: OpenRouter (Cloud API)
    echo     - Get an API key from: https://openrouter.ai/
    echo     - Configure in Settings tab after launch
    echo.
    echo The system will start - configure your preferred provider in Settings.
) else (
    echo LM Studio is running and responding!
)
echo.

REM ================================================================
REM STEP 7: Clean Up Existing Processes
REM ================================================================
echo [7/8] Cleaning up existing processes on ports 8000 and 5173...
echo.

REM Kill any process using port 8000 (backend)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo Found process %%a using port 8000, terminating...
    taskkill /F /PID %%a >nul 2>&1
)

REM Kill any process using port 5173 (frontend)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING') do (
    echo Found process %%a using port 5173, terminating...
    taskkill /F /PID %%a >nul 2>&1
)

REM Wait a moment for ports to be released
timeout /t 2 /nobreak >nul

echo Ports cleaned successfully!
echo.

REM Verify ports are now available
netstat -ano | findstr :8000 | findstr LISTENING >nul 2>&1
if not errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Port 8000 is still in use after cleanup attempt!
    echo ============================================================
    echo.
    echo Please manually close any applications using port 8000.
    echo You can check with: netstat -ano ^| findstr :8000
    echo.
    pause
    exit /b 1
)

netstat -ano | findstr :5173 | findstr LISTENING >nul 2>&1
if not errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Port 5173 is still in use after cleanup attempt!
    echo ============================================================
    echo.
    echo Please manually close any applications using port 5173.
    echo You can check with: netstat -ano ^| findstr :5173
    echo.
    pause
    exit /b 1
)

echo Ports 8000 and 5173 are available!
echo.

REM ================================================================
REM STEP 8: Start Services
REM ================================================================
echo [8/8] Starting services...
echo.
echo ================================================================
echo   SYSTEM STARTING
echo ================================================================
echo.
echo Backend API will run on: http://localhost:8000
echo Frontend UI will run on: http://localhost:5173
echo.
echo Two windows will open:
echo   - ASI Backend (Keep this window open)
echo   - ASI Frontend (Keep this window open)
echo.
echo Starting services automatically in 3 seconds... (Ctrl+C to cancel)
timeout /t 3 /nobreak >nul
echo.

REM Start backend in separate window with proper path
echo Starting backend server...
start "ASI Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --no-access-log"

REM Wait a few seconds for backend to start
echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Start frontend in separate window
echo Starting frontend server...
start "ASI Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

REM Wait for frontend to initialize
echo Waiting for frontend to initialize...
timeout /t 8 /nobreak >nul

REM Open browser automatically
echo Opening browser...
start http://localhost:5173

echo.
echo ================================================================
echo   SYSTEM STARTED!
echo ================================================================
echo.
echo Two windows have opened:
echo   - ASI Backend (running on port 8000)
echo   - ASI Frontend (running on port 5173)
echo.
echo Browser opened automatically to: http://localhost:5173
echo If it didn't open, open that URL manually.
echo.
echo To stop the system: Close both service windows
echo.
echo This launcher window can now be closed.
echo.
echo Closing launcher window automatically in 3 seconds...
timeout /t 3 /nobreak >nul
exit /b 0
