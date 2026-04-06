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
4. Формат одной строки JSONL:

```json
{"image": "000123.jpg", "question": "Что изображено?", "answer": "Красный автобус"}
```

Поддерживаются также ключи `image_path`, `image_file`, `image_filename`.

Если нужен автоматический импорт из Hugging Face:

```bash
python3 scripts/download_gqa_ru_snapshot.py \
  --output-dir data/raw/hf_gqa_ru_snapshot

python3 scripts/prepare_gqa_ru_dataset.py \
  --output-root data/gqa_ru \
  --source-dir data/raw/hf_gqa_ru_snapshot \
  --val-size 0.05 \
  --answer-field answer
```

Источник: `deepvk/GQA-ru`. Первый скрипт скачивает весь dataset repository, второй
читает parquet локально, сохраняет картинки и делит train/val по `imageId`.

## 3) Обучение

```bash
./scripts/train.sh \
  --config configs/experiments/gqa_ru_qwen25vl_lora_v1.yaml \
  --device cuda:0 \
  --output-dir outputs/gqa_ru_train
```

Что делает скрипт:
- загружает experiment-конфиг и связанные `data/model/train/eval` YAML;
- поднимает `Qwen/Qwen2.5-VL-3B-Instruct` через `transformers`;
- применяет LoRA из `configs/train/lora_ft.yaml`;
- сохраняет чекпоинты в `--output-dir` и метрики в `runs/<experiment>/train_metrics.json`.

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
