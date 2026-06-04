#!/usr/bin/env bash
set -euo pipefail

QWEN35_BASE="${QWEN35_BASE:-Qwen/Qwen3.5-0.8B}"
QWEN35_ADAPTER="${QWEN35_ADAPTER:-/home/lockr/projects/VK_education_vllm/checkpoints/gqa_ru_qwen35_0_8b_lora_fast_v1}"
QWEN35_MERGED="${QWEN35_MERGED:-artifacts/merged_qwen35_gqa_ru_full}"

QWEN25_BASE="${QWEN25_BASE:-Qwen/Qwen2.5-VL-3B-Instruct}"
QWEN25_ADAPTER="${QWEN25_ADAPTER:-checkpoints/gqa_ru_qwen25vl_lora_full_v1}"
QWEN25_MERGED="${QWEN25_MERGED:-artifacts/merged_qwen25vl_gqa_ru_full}"
QWEN25_CONFIG="${QWEN25_CONFIG:-configs/experiments/gqa_ru_qwen25vl_lora_full_v1.yaml}"

run_eval() {
  local model_backend="$1"
  local pretrained="$2"
  local output_path="$3"
  shift 3

  uv run python -m lmms_eval eval \
    --model "${model_backend}" \
    --model_args "pretrained=${pretrained}$*" \
    --tasks gqa-ru \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}" \
    --device cuda:0 \
    --verbosity INFO \
    --trust_remote_code
}

merge_adapter() {
  local base="$1"
  local adapter="$2"
  local output="$3"

  uv run python - "${base}" "${adapter}" "${output}" <<'PY'
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base, adapter, output = sys.argv[1:4]
output_path = Path(output)
output_path.mkdir(parents=True, exist_ok=True)

model = AutoModelForImageTextToText.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(output_path, safe_serialization=True, max_shard_size="2GB")
AutoProcessor.from_pretrained(base, trust_remote_code=True).save_pretrained(output_path)
PY
}

mkdir -p runs/lmms_eval artifacts checkpoints

if [[ ! -f "${QWEN35_MERGED}/config.json" ]]; then
  merge_adapter "${QWEN35_BASE}" "${QWEN35_ADAPTER}" "${QWEN35_MERGED}"
fi

run_eval qwen3_5 "${QWEN35_BASE}" runs/lmms_eval/gqa_ru_qwen35_base_full ",enable_thinking=False"
run_eval qwen3_5 "${QWEN35_MERGED}" runs/lmms_eval/gqa_ru_qwen35_lora_full ",enable_thinking=False"

uv run python scripts/train.py \
  --config "${QWEN25_CONFIG}" \
  --device cuda:0 \
  --output-dir "${QWEN25_ADAPTER}"

merge_adapter "${QWEN25_BASE}" "${QWEN25_ADAPTER}" "${QWEN25_MERGED}"
run_eval qwen2_5_vl "${QWEN25_BASE}" runs/lmms_eval/gqa_ru_qwen25vl_base_full
run_eval qwen2_5_vl "${QWEN25_MERGED}" runs/lmms_eval/gqa_ru_qwen25vl_lora_full
