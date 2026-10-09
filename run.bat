@echo off
echo ===================================================
echo Auto-PCOS Classification Application Starter
echo ===================================================

echo Starting FastAPI Backend...
start cmd /k "python api.py"

echo Starting React Frontend...
cd frontend
start cmd /k "npm run dev"

echo Both services are starting up!
echo The React Dashboard will be available at: http://localhost:5173
echo The API will be running at: http://localhost:8000
echo.
echo Press any key to close this launcher window...
pause >nul
