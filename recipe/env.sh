#!/usr/bin/env bash
# Qwopus3.6-27B-v2-MTP Q4_K_M — environment for DGX Spark (GB10)
export MODEL_PATH=/home/$USER/models/qwopus/Qwopus3.6-27B-v2-MTP-Q4_K_M.gguf
export LLAMA_SERVER=/home/$USER/llama.cpp/build-cuda/bin/llama-server
export HOST=0.0.0.0
export PORT=8000
export ENDPOINT=http://localhost:${PORT}/v1
export CONTEXT_SIZE=262144
export KV_TYPE=q8_0
export SPEC_TYPE=draft-mtp
export SPEC_DRAFT_N_MAX=3
export THREADS=20
