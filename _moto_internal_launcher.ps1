# MOTO Internal Launcher (PowerShell)
# This is an internal script. Use "Press to Launch MOTO.bat" instead.
# If needed manually: powershell -ExecutionPolicy Bypass -File _moto_internal_launcher.ps1

# ================================================================
# CRITICAL: This prevents the window from closing on errors
# ================================================================
$ErrorActionPreference = "Stop"

function Exit-WithPause {
    param([int]$ExitCode = 0)
    Write-Host ""
    Write-Host "Press Enter to close..." -ForegroundColor Yellow
    Read-Host
    exit $ExitCode
}

try {
    Clear-Host
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  ASI Aggregator-Compiler System - One-Click Launcher" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""

    # Function to check if command exists
    function Test-Command($cmdname) {
        return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
    }

    # Check for Python
    Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
    if (-not (Test-Command python)) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install Python 3.8+ from:" -ForegroundColor Yellow
        Write-Host "https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "IMPORTANT: Check 'Add Python to PATH' during installation" -ForegroundColor Yellow
        Exit-WithPause -ExitCode 1
    }
    $pythonVersion = python --version
    Write-Host $pythonVersion -ForegroundColor Green
    Write-Host ""

    # Check for Node.js
    Write-Host "[2/6] Checking Node.js installation..." -ForegroundColor Yellow
    if (-not (Test-Command node)) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host "ERROR: Node.js is not installed or not in PATH" -ForegroundColor Red
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install Node.js 16+ from:" -ForegroundColor Yellow
        Write-Host "https://nodejs.org/" -ForegroundColor Yellow
        Exit-WithPause -ExitCode 1
    }
    $nodeVersion = node --version
    $npmVersion = npm --version
    Write-Host "Node: $nodeVersion" -ForegroundColor Green
    Write-Host "npm: $npmVersion" -ForegroundColor Green
    Write-Host ""

    # Create necessary directories
    Write-Host "[3/6] Creating necessary directories..." -ForegroundColor Yellow
    $directories = @(
        "backend\data",
        "backend\data\user_uploads",
        "backend\logs"
    )
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "Created: $dir" -ForegroundColor Green
        }
    }
    Write-Host "Directories ready!" -ForegroundColor Green
    Write-Host ""

    # Check/Install Python dependencies
    Write-Host "[4/6] Checking Python dependencies..." -ForegroundColor Yellow
    $pipList = pip list 2>&1
    if ($pipList -notmatch "fastapi") {
        Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
        Write-Host "This may take a few minutes..." -ForegroundColor Yellow
        Write-Host ""
        pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "============================================================" -ForegroundColor Red
            Write-Host "ERROR: Failed to install Python dependencies" -ForegroundColor Red
            Write-Host "============================================================" -ForegroundColor Red
            Write-Host ""
            Write-Host "Please check:" -ForegroundColor Yellow
            Write-Host "- Internet connection is working" -ForegroundColor Yellow
            Write-Host "- You have permission to install packages" -ForegroundColor Yellow
            Exit-WithPause -ExitCode 1
        }
        Write-Host "Python dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "Python dependencies already installed" -ForegroundColor Green
    }
    Write-Host ""

    # Check/Install Node.js dependencies
    Write-Host "[5/6] Checking Node.js dependencies..." -ForegroundColor Yellow
    if (-not (Test-Path "frontend")) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host "ERROR: frontend directory not found!" -ForegroundColor Red
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "Make sure you're running this from the project root directory." -ForegroundColor Yellow
        Exit-WithPause -ExitCode 1
    }
    
    Set-Location frontend
    if (-not (Test-Path "node_modules")) {
        Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
        Write-Host "This may take a few minutes..." -ForegroundColor Yellow
        Write-Host ""
        npm install
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "============================================================" -ForegroundColor Red
            Write-Host "ERROR: Failed to install Node.js dependencies" -ForegroundColor Red
            Write-Host "============================================================" -ForegroundColor Red
            Write-Host ""
            Write-Host "Please check:" -ForegroundColor Yellow
            Write-Host "- Internet connection is working" -ForegroundColor Yellow
            Write-Host "- package.json exists in frontend directory" -ForegroundColor Yellow
            Set-Location ..
            Exit-WithPause -ExitCode 1
        }
        Write-Host "Node.js dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "Node.js dependencies already installed" -ForegroundColor Green
    }
    Write-Host "Fixing known vulnerabilities..." -ForegroundColor Yellow
    npm audit fix 2>&1 | Out-Null
    Set-Location ..
    Write-Host ""

    # Check for LM Studio (optional - OpenRouter is an alternative)
    Write-Host "[6/6] Checking LM Studio..." -ForegroundColor Yellow
    Write-Host ""
    
    # Check if LM Studio is responding
    $lmStudioAvailable = $false
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:1234/v1/models" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        $lmStudioAvailable = $true
    } catch {
        $lmStudioAvailable = $false
    }
    
    if ($lmStudioAvailable) {
        Write-Host "LM Studio is running and responding!" -ForegroundColor Green
    } else {
        Write-Host "================================================================" -ForegroundColor Cyan
        Write-Host "NOTE: LM Studio is not detected on http://127.0.0.1:1234" -ForegroundColor Cyan
        Write-Host "================================================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "This is OK! You have two options for AI models:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Option 1: LM Studio (Local)" -ForegroundColor Yellow
        Write-Host "    - Download from: https://lmstudio.ai/" -ForegroundColor White
        Write-Host "    - Load a model and start the Local Server" -ForegroundColor White
        Write-Host ""
        Write-Host "  Option 2: OpenRouter (Cloud API)" -ForegroundColor Yellow
        Write-Host "    - Get an API key from: https://openrouter.ai/" -ForegroundColor White
        Write-Host "    - Configure in Settings tab after launch" -ForegroundColor White
        Write-Host ""
        Write-Host "The system will start - configure your preferred provider in Settings." -ForegroundColor Green
    }
    Write-Host ""

    # Clean up any existing processes on ports 8000 and 5173
    Write-Host "[7/7] Cleaning up existing processes on ports 8000 and 5173..." -ForegroundColor Yellow
    Write-Host ""
    
    # Kill processes on port 8000
    $port8000 = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if ($port8000) {
        foreach ($conn in $port8000) {
            Write-Host "Found process $($conn.OwningProcess) using port 8000, terminating..." -ForegroundColor Yellow
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
    
    # Kill processes on port 5173
    $port5173 = Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue
    if ($port5173) {
        foreach ($conn in $port5173) {
            Write-Host "Found process $($conn.OwningProcess) using port 5173, terminating..." -ForegroundColor Yellow
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
    
    # Wait for ports to be released
    Start-Sleep -Seconds 2
    
    # Verify ports are available
    $port8000Check = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if ($port8000Check) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host "ERROR: Port 8000 is still in use after cleanup attempt!" -ForegroundColor Red
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please manually close any applications using port 8000." -ForegroundColor Yellow
        Write-Host "You can check with: Get-NetTCPConnection -LocalPort 8000" -ForegroundColor Yellow
        Exit-WithPause -ExitCode 1
    }
    
    $port5173Check = Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue
    if ($port5173Check) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host "ERROR: Port 5173 is still in use after cleanup attempt!" -ForegroundColor Red
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please manually close any applications using port 5173." -ForegroundColor Yellow
        Write-Host "You can check with: Get-NetTCPConnection -LocalPort 5173" -ForegroundColor Yellow
        Exit-WithPause -ExitCode 1
    }
    
    Write-Host "Ports 8000 and 5173 are available!" -ForegroundColor Green
    Write-Host ""

    # Start the system
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  All checks passed! Starting system..." -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Backend API will run on: http://localhost:8000" -ForegroundColor Green
    Write-Host "Frontend UI will run on: http://localhost:5173" -ForegroundColor Green
    Write-Host ""
    Write-Host "Press any key to start the services..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    Write-Host ""

    # Start backend in new window
    $backendPath = Join-Path $PSScriptRoot "backend"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; Write-Host 'Starting Backend...' -ForegroundColor Cyan; python -m api.main"

    # Wait for backend to start
    Write-Host "Waiting for backend to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5

    # Start frontend in new window
    $frontendPath = Join-Path $PSScriptRoot "frontend"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; Write-Host 'Starting Frontend...' -ForegroundColor Cyan; npm run dev"

    # Wait for frontend to initialize
    Write-Host "Waiting for frontend to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 8

    # Open browser automatically
    Write-Host "Opening browser..." -ForegroundColor Green
    Start-Process "http://localhost:5173"

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  SYSTEM STARTED!" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Two windows have opened:" -ForegroundColor Green
    Write-Host "  - ASI Backend (running on port 8000)" -ForegroundColor Green
    Write-Host "  - ASI Frontend (running on port 5173)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Browser opened automatically to:" -ForegroundColor Green
    Write-Host "  http://localhost:5173" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "If it didn't open, open that URL manually." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To stop the system: Close both service windows" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "This launcher window can now be closed." -ForegroundColor Green
    Write-Host ""
    Exit-WithPause -ExitCode 0

} catch {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "FATAL ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Stack Trace:" -ForegroundColor Yellow
    Write-Host $_.ScriptStackTrace -ForegroundColor Yellow
    Exit-WithPause -ExitCode 1
}
