@echo off
REM Activate virtual environment on Windows

if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo Virtual environment activated!
    echo.
    echo You can now run your Python scripts.
    echo To deactivate, type: deactivate
    cmd /k
) else if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo Virtual environment activated!
    echo.
    echo You can now run your Python scripts.
    echo To deactivate, type: deactivate
    cmd /k
) else (
    echo Error: Virtual environment not found!
    echo.
    echo To create one, run:
    echo   python -m venv venv
    echo.
    pause
)

