# GF-1 Cloud vs Non-Cloud Pipeline (MVP)

This workspace contains an end-to-end implementation of the pipeline described in `gf1_cloud_llm_pipeline.md` for the GF-1 WFV dataset at `/home/data/cloud_detec_dataset/GF_1`.

## Dataset Assumptions
- Image patches: `.npy`, shape `(512, 512, 4)` with channels `(B, G, R, NIR)`.
- Mask patches: `.npy`, shape `(512, 512)` with values:
  - `0`: non-cloud
  - `255`: cloud
  - `254`: shadow (treated as non-cloud in this refactor)

If your mask encoding differs, update `gf1_cloud/data.py`.

## Project Layout
- `gf1_cloud/`: core modules (dataset, model, features, router, fusion, patch sampling).
- `configs/`: policy, fusion, and calibration defaults.
- `scripts/`: train, calibrate thresholds, build features, run pipeline.

## Quick Start
Set the data root (your dataset path):

```bash
export DATA_ROOT=/home/data/cloud_detec_dataset/GF_1
export MODEL_DIR=/home/data/KXShen/model/gf1_cloud
```

Install progress bar dependency:
```bash
conda install tqdm -y
```

Start vLLM (required). This script will download/cache the LLM under `$MODEL_DIR/hf_cache`:
```bash
bash scripts/start_vllm.sh
```

### 1) Train CloudSegNet (UNet baseline)
```bash
python scripts/train_seg.py --data-root $DATA_ROOT --out-dir $MODEL_DIR/seg
```

默认使用 GPU（`--device cuda`），若 CUDA 不可用会直接报错。

### 2) Calibrate thresholds (t_cloud)
```bash
python scripts/calibrate_thresholds.py \
  --data-root $DATA_ROOT \
  --checkpoint $MODEL_DIR/seg/cloudsegnet_best.pt \
  --out outputs/calibration.json
```

### 3) Build scene features + route decisions (FAST/ESCALATE)
```bash
python scripts/build_features.py \
  --data-root $DATA_ROOT \
  --checkpoint $MODEL_DIR/seg/cloudsegnet_best.pt \
  --calibration outputs/calibration.json \
  --out-dir outputs/features \
  --llm-api-url http://localhost:8000/v1/chat/completions \
  --llm-model Qwen/Qwen2-VL-7B-Instruct-AWQ
```

### 4) Run full pipeline (router + fusion + patch sampling)
```bash
python scripts/run_pipeline.py \
  --data-root $DATA_ROOT \
  --checkpoint $MODEL_DIR/seg/cloudsegnet_best.pt \
  --calibration outputs/calibration.json \
  --policy configs/policy_v1.json \
  --fusion configs/fusion_weights.json \
  --out-dir outputs/pipeline \
  --patch-dir outputs/patches \
  --mask-dir outputs/masks \
  --batch-size 4 \
  --num-workers 4 \
  --llm-api-url http://localhost:8000/v1/chat/completions \
  --llm-model Qwen/Qwen2-VL-7B-Instruct-AWQ
```

## Notes
- Router and patch verifier are LLM-driven and required for the full pipeline.
- The fusion is a weighted score with configurable thresholds (`configs/fusion_weights.json`).
- Use `PYTHONPATH=.` if your environment does not auto-resolve local imports.

## Standalone LLM Utilities
Run LLM router on feature JSON files:
```bash
python scripts/run_llm_router.py \
  --features outputs/features \
  --model Qwen/Qwen2-VL-7B-Instruct-AWQ \
  --out-dir outputs/llm_router
```

Run patch verifier on saved patch images:
```bash
python scripts/run_llm_patch_verifier.py \
  --patch-dir outputs/patches \
  --model Qwen/Qwen2-VL-7B-Instruct-AWQ \
  --out-dir outputs/patch_verify
```
