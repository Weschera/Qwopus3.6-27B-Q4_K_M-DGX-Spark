#!/usr/bin/env bash
# Qwopus3.6-27B-v2-MTP Q4_K_M — launch llama-server on DGX Spark (GB10)
# Run after: source env.sh
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH in env.sh}"
: "${LLAMA_SERVER:?set LLAMA_SERVER in env.sh}"

"${LLAMA_SERVER}" \
  -m "${MODEL_PATH}" \
  -ngl 99 \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --jinja \
  -c "${CONTEXT_SIZE:-262144}" \
  -np 1 \
  -fa on \
  -ctk "${KV_TYPE:-q8_0}" \
  -ctv "${KV_TYPE:-q8_0}" \
  --spec-type "${SPEC_TYPE:-draft-mtp}" \
  --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-3}" \
  -t "${THREADS:-20}"

# Smoke test:
#   curl http://<spark-ip>:8000/v1/chat/completions \
#     -H 'Content-Type: application/json' \
#     -d '{"model":"Qwopus3.6-27B-v2-MTP-Q4_K_M.gguf","messages":[{"role":"user","content":"Reply exactly: OK"}],"max_tokens":16,"temperature":0}'
