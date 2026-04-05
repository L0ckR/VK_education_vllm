# Runbook: запуск на GPU-сервере

Ниже базовые команды для обучения/оценки VLM в этом репозитории.

## 1) Подготовка окружения

```bash
cd /path/to/VK_education_vllm
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## 2) Подготовка данных

1. Скачайте открытые датасеты из коллекции VK/DeepVK на Hugging Face.
2. Подготовьте манифесты в формате JSONL:
   - `data/gqa_ru/train.jsonl`, `val.jsonl`, `test.jsonl`
   - `data/mmbench_ru/train.jsonl`, `val.jsonl`, `test.jsonl`
3. Разместите изображения в соответствующих `images/` директориях.

## 3) Обучение

```bash
./scripts/train.sh \
  --config configs/experiments/gqa_ru_qwen25vl_lora_v1.yaml \
  --device cuda:0 \
  --output-dir outputs/gqa_ru_train
```

## 4) Оценка

```bash
./scripts/eval.sh \
  --config configs/experiments/gqa_ru_qwen25vl_lora_v1.yaml \
  --device cuda:0 \
  --output-dir outputs/gqa_ru_eval \
  --resume checkpoints/gqa_ru_qwen25vl_lora_v1/best.ckpt
```

## 5) Предсказание

```bash
./scripts/predict.sh \
  --config configs/experiments/gqa_ru_qwen25vl_lora_v1.yaml \
  --device cuda:0 \
  --output-dir outputs/gqa_ru_predict \
  --resume checkpoints/gqa_ru_qwen25vl_lora_v1/best.ckpt
```

## 6) Проверка согласованности

```bash
python scripts/check_repo_consistency.py
```

Проверяет структуру проекта и валидность ссылок на конфиги в `configs/experiments`.
