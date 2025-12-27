#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/data/KXShen/model/gf1_cloud}"
CACHE_DIR="${HF_HOME:-$MODEL_DIR/hf_cache}"

export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"

mkdir -p "$CACHE_DIR"

MODEL="${VLLM_MODEL:-Qwen/Qwen2-VL-7B-Instruct-AWQ}"

# If MODEL is a repo name and a local snapshot exists, prefer the first snapshot path.
if [[ ! -d "$MODEL" ]]; then
  snapshot_path=$(find "$CACHE_DIR" -path "*Qwen2-VL-7B-Instruct-AWQ*/snapshots/*" -maxdepth 6 -type d | head -n1)
  if [[ -n "$snapshot_path" ]]; then
    MODEL="$snapshot_path"
    echo "Using local snapshot: $MODEL"
  else
    echo "No local snapshot found, will try to download: $MODEL"
  fi
fi

exec vllm serve "$MODEL" \
  --dtype auto \
  --max-model-len 8192 \
  --allowed-local-media-path /home/ps/KXShen/syncfolder/LLM/outputs
