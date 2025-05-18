@echo off
echo DocuMorph AI Server with MongoDB Fallback
echo ----------------------------------------

:: Set environment variables
set PYTHONPATH=%PYTHONPATH%;%cd%

:: Try to connect to MongoDB
echo Testing MongoDB connection...
cd server
python test_mongodb.py
if %ERRORLEVEL% NEQ 0 (
  echo MongoDB connection failed. Running in fallback mode...
  python run_server.py
) else (
  echo MongoDB connection successful. Starting server normally...
  cd server
  python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
)

echo.
echo Server terminated.
pause 