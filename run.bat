@echo off
setlocal

cd /d "%~dp0"

echo Installing Python dependencies...
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo Python 3.10+ is required. Please install Python 3.10 or newer.
  exit /b 1
)
if "%PIP_INDEX_URL%"=="" set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
if "%PIP_TRUSTED_HOST%"=="" set PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
if "%HF_ENDPOINT%"=="" set HF_ENDPOINT=https://hf-mirror.com
python -m pip install -r requirements.txt -i "%PIP_INDEX_URL%" --trusted-host "%PIP_TRUSTED_HOST%" --timeout 120 --retries 10

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo .env has been created from .env.example.
  echo Please edit .env and set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL before asking questions.
)

echo Cleaning old processes on ports 8000 and 8501...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /PID %%a /F >nul 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501') do taskkill /PID %%a /F >nul 2>nul

if not exist "output" mkdir output
if not exist "uploads" mkdir uploads
if not exist "chroma_data" mkdir chroma_data

echo Starting FastAPI backend on http://localhost:8000 ...
start "ScholarQA Backend" /min cmd /c "python backend\main.py > output\backend.log 2>&1"

echo Waiting for FastAPI backend to become ready...
for /l %%i in (1,1,60) do (
  powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health | Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
  if not errorlevel 1 (
    echo Backend is ready.
    goto backend_ready
  )
  timeout /t 1 /nobreak >nul
)

echo Backend did not become ready in 60 seconds. Please check output\backend.log.
exit /b 1

:backend_ready
echo Open your browser at: http://localhost:8501
echo Starting Streamlit frontend...
python -m streamlit run app.py --server.port 8501

endlocal
