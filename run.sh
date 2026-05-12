#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

kill_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti:"$port" | xargs kill -9 2>/dev/null || true
  fi
}

wait_backend() {
  echo "Waiting for FastAPI backend to become ready..."
  for attempt in {1..60}; do
    if curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
      echo "Backend is ready."
      return 0
    fi
    sleep 1
  done

  echo "Backend did not become ready in 60 seconds. Recent backend log:"
  tail -n 80 output/backend.log 2>/dev/null || true
  return 1
}

echo "Installing Python dependencies..."
python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 'Python 3.10+ is required. Please install Python 3.10 or newer.')"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
python3 -m pip install -r requirements.txt \
  -i "${PIP_INDEX_URL}" \
  --trusted-host "${PIP_TRUSTED_HOST}" \
  --timeout 120 \
  --retries 10

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ".env has been created from .env.example."
  echo "Please edit .env and set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL before asking questions."
fi

echo "Cleaning old processes on ports 8000 and 8501..."
kill_port 8000
kill_port 8501

mkdir -p output uploads chroma_data

echo "Starting FastAPI backend on http://localhost:8000 ..."
python3 backend/main.py > output/backend.log 2>&1 &
BACKEND_PID=$!
trap 'kill ${BACKEND_PID} 2>/dev/null || true' EXIT
echo "Backend PID: ${BACKEND_PID}"

wait_backend

echo "Open your browser at: http://localhost:8501"
echo "Starting Streamlit frontend..."
python3 -m streamlit run app.py --server.port 8501
