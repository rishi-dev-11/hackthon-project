@echo off
ECHO ======= DocuMorph AI Development Startup =======
ECHO.

:: Check for dependencies
WHERE python >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO Python not found! Please install Python 3.9+ and try again.
    EXIT /B 1
)

WHERE npm >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO Node.js/npm not found! Please install Node.js and try again.
    EXIT /B 1
)

:: Create .env file if it doesn't exist
IF NOT EXIST .env (
    ECHO Creating default .env file...
    (
        ECHO # MongoDB Connection
        ECHO MONGO_URI=mongodb://localhost:27017/documorph
        ECHO.
        ECHO # App Settings
        ECHO APP_ENV=development
        ECHO JWT_SECRET=supersecretkey
        ECHO JWT_ALGORITHM=HS256
        ECHO JWT_EXPIRATION_MINUTES=1440
        ECHO.
        ECHO # API Keys
        ECHO LANGCHAIN_PROJECT=
        ECHO HF_TOKEN=
        ECHO groq_api_key=
        ECHO serpapi=
        ECHO GOOGLE_CLIENT_ID=
        ECHO Client_secret=
        ECHO goog_api_key=
        ECHO GOOGLE_REDIRECT_URI=
    ) > .env
    ECHO .env file created with default values. Please update with your actual API keys.
) ELSE (
    ECHO .env file already exists.
)

:: Ensure dist directory exists to prevent backend errors
IF NOT EXIST frontend\dist (
    ECHO Creating frontend\dist directory...
    MKDIR frontend\dist
    ECHO. > frontend\dist\.placeholder
)

:: Start backend server in a new window
ECHO Starting backend server...
start cmd /k "cd server && python -m pip install -r requirements.txt && python -m uvicorn main:app --reload"

:: Install frontend dependencies and start frontend in a new window
ECHO Starting frontend...
start cmd /k "cd frontend && npm install && npm run dev"

ECHO.
ECHO ======= Startup Complete =======
ECHO Backend server running at: http://localhost:8000
ECHO Frontend server running at: http://localhost:5173
ECHO.
ECHO Press any key to close this window. The servers will continue running.
PAUSE > NUL 