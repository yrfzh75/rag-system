#!/usr/bin/env bash
set -euo pipefail

if [[ -x .venv/bin/python ]]; then
  validation_python=.venv/bin/python
else
  validation_python=python3
fi

"$validation_python" -m pytest -q
"$validation_python" -m ruff check .
"$validation_python" -m rag_mvp.evaluation.cli \
  --dataset evals/retrieval_cases.json --output evals/reports/retrieval_latest --top-k 3

curl --silent --fail http://127.0.0.1:11434/api/tags >/dev/null || {
  echo "Ollama is required: start Ollama and rerun ./scripts/evaluate.sh" >&2
  exit 1
}

validation_log="artifacts/qa_validation_events.jsonl"
mkdir -p artifacts
rm -f "$validation_log"
QA_CACHE_ENABLED=false QA_LOG_PATH="$validation_log" \
  "$validation_python" -m uvicorn rag_mvp.main:app --host 127.0.0.1 --port 8010 >artifacts/validation_api.log 2>&1 &
api_pid=$!
trap 'kill "$api_pid" 2>/dev/null || true' EXIT

for _ in {1..30}; do
  curl --silent --fail http://127.0.0.1:8010/health >/dev/null && break
  sleep 1
done
curl --silent --fail http://127.0.0.1:8010/health >/dev/null || {
  echo "Validation API failed to start; inspect artifacts/validation_api.log" >&2
  exit 1
}

"$validation_python" -m rag_mvp.evaluation.qa \
  --endpoint http://127.0.0.1:8010/qa \
  --dataset evals/qa_cases.json --output evals/reports/qa_latest
"$validation_python" -m rag_mvp.evaluation.load \
  --endpoint http://127.0.0.1:8010/qa --concurrency 5 --requests 5 --expect-cold-cache
"$validation_python" -m rag_mvp.evaluation.operations \
  --logs "$validation_log" --qa-results evals/reports/qa_latest/results.json \
  --output artifacts/operations.csv
