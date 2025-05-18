@echo off
:: DocuMorph AI Installation Script for Windows
echo ===============================================================
echo =            DocuMorph AI Installation Script                 =
echo ===============================================================
echo.

:: Check if Python 3.8+ is installed
python --version 2>NUL
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher.
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)
echo [SUCCESS] Created virtual environment

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)
echo [SUCCESS] Activated virtual environment

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [ERROR] Failed to upgrade pip.
    exit /b 1
)
echo [SUCCESS] Upgraded pip

:: Install requirements
echo Installing requirements (this may take a while)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements.
    exit /b 1
)
echo [SUCCESS] Installed requirements

:: Install spaCy model
echo Installing spaCy model...
python -m spacy download en_core_web_sm
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download spaCy model.
    exit /b 1
)
echo [SUCCESS] Installed spaCy model

echo.
echo [SUCCESS] Installation complete!
echo.
echo To start DocuMorph AI, run:
echo venv\Scripts\activate
echo cd backend
echo streamlit run documorph_ai.py
echo.
echo ===============================================================

pause 