@echo off
cd /d "%~dp0"
py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo Install Python 3.13, then run setup.cmd again.
    pause
    exit /b 1
)
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -c "import sys; sys.exit(sys.version_info[:2] != (3, 13))"
    if errorlevel 1 (
        echo Existing .venv uses another Python version.
        echo Rename .venv to .venv-old, then run setup.cmd again.
        pause
        exit /b 1
    )
)
py -3.13 -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo Setup complete.
pause
