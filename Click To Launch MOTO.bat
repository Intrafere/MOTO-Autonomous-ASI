@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "LAUNCHER_SCRIPT=%SCRIPT_DIR%moto_launcher.py"
set "PYTHON_CMD="

call :find_python
if not defined PYTHON_CMD (
    echo.
    echo Python 3.10+ was not found. MOTO will try to install Python 3.12 with winget.
    echo.
    call :install_python
    if errorlevel 1 goto python_missing
    call :find_python
)

if not defined PYTHON_CMD goto python_missing

%PYTHON_CMD% "%LAUNCHER_SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"
if %EXIT_CODE% NEQ 0 (
    echo.
    echo Press Enter to close...
    pause >nul
)
exit /b %EXIT_CODE%

:find_python
call :check_python py -3.12
if defined PYTHON_CMD exit /b 0
call :check_python py -3.11
if defined PYTHON_CMD exit /b 0
call :check_python py -3.10
if defined PYTHON_CMD exit /b 0
call :check_python "%LocalAppData%\Programs\Python\Python312\python.exe"
if defined PYTHON_CMD exit /b 0
call :check_python "%LocalAppData%\Programs\Python\Python311\python.exe"
if defined PYTHON_CMD exit /b 0
call :check_python "%LocalAppData%\Programs\Python\Python310\python.exe"
if defined PYTHON_CMD exit /b 0
call :check_python "%ProgramFiles%\Python312\python.exe"
if defined PYTHON_CMD exit /b 0
call :check_python "%ProgramFiles%\Python311\python.exe"
if defined PYTHON_CMD exit /b 0
call :check_python "%ProgramFiles%\Python310\python.exe"
if defined PYTHON_CMD exit /b 0
call :check_python "%ProgramFiles(x86)%\Python312\python.exe"
if defined PYTHON_CMD exit /b 0
call :check_python "%ProgramFiles(x86)%\Python311\python.exe"
if defined PYTHON_CMD exit /b 0
call :check_python "%ProgramFiles(x86)%\Python310\python.exe"
if defined PYTHON_CMD exit /b 0
call :check_python python
if defined PYTHON_CMD exit /b 0
call :check_python py -3
if defined PYTHON_CMD exit /b 0
call :check_python "%LocalAppData%\Programs\Python\Python313\python.exe"
if defined PYTHON_CMD exit /b 0
call :check_python "%ProgramFiles%\Python313\python.exe"
if defined PYTHON_CMD exit /b 0
call :check_python "%ProgramFiles(x86)%\Python313\python.exe"
if defined PYTHON_CMD exit /b 0
exit /b 1

:check_python
%* -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=%*"
exit /b 0

:install_python
where winget >nul 2>nul
if errorlevel 1 exit /b 1
winget install --id Python.Python.3.12 -e --source winget --scope user --accept-package-agreements --accept-source-agreements
if not errorlevel 1 exit /b 0
echo.
echo User-scope Python install did not complete. Trying the default winget install scope...
winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements
exit /b %ERRORLEVEL%

:python_missing
echo.
echo ============================================================
echo ERROR: Python 3.10+ is required to launch MOTO.
echo ============================================================
echo.
echo Automatic Python installation was unavailable or did not complete.
echo Install Python 3.12 from https://www.python.org/downloads/
echo IMPORTANT: Check "Add Python to PATH" during installation.
echo.
start "" "https://www.python.org/downloads/"
echo Press Enter to close...
pause >nul
exit /b 1
