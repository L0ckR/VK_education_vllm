# VK Education: Vision-Language Modeling проект

**HF artifact:** https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen35-08b-lora
**Best run:** `gqa_ru_qwen35_0_8b_lora_fast_v1`, LoRA adapter, best checkpoint `checkpoint-4560`,
`eval_loss=0.4337001144886017`.

Репозиторий подготовлен под проект VK Education по обучению и оценке VLM-моделей на открытых
данных VK (коллекция DeepVK на Hugging Face), включая бенчмарки **GQA-ru** и **MMBench-ru**.

## Цель репозитория

Дать воспроизводимый проект, где:
- основной Python-код и утилиты находятся в `src/`;
- запуск выполняется через скрипты в `scripts/`;
- параметры экспериментов зафиксированы в `configs/`;
- описание проекта, модели и артефактов оформлено в `docs/` и `reports/`;
- структура для данных и исследований (EDA) выделена в `data/` и `notebooks/`.

## Структура

- `src/vk_vlm_project/` — пакет с базовыми классами/утилитами и CLI entrypoints.
- `configs/model/` — параметры VLM-модели и процессора.
- `configs/data/` — описания датасетов (GQA-ru, MMBench-ru).
- `configs/train/` — параметры обучения (LoRA fine-tuning).
- `configs/eval/` — параметры оценки.
- `configs/inference/` — параметры инференса.
- `configs/experiments/` — experiment-level YAML, объединяющие model/data/train/eval.
- `scripts/` — скрипты запуска train/eval/predict и проверок согласованности.
- `docs/` — постановка проекта, runbook, шаблон и подробное описание решения.
- `reports/` — итоговый отчет и machine-readable summary метрик.
- `data/` — структура для манифестов/примеров без тяжёлых артефактов.
- `notebooks/eda/` — место для исследовательских ноутбуков и EDA.

## Итоговый артефакт

Опубликован LoRA-адаптер:
https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen35-08b-lora

Автор проекта: Ибрагимов Далгат Магомедалиевич, МАИ институт 8, группа М8О-308Б-32.

Фактический запуск:
- данные: `deepvk/GQA-ru`, подготовлены в локальные JSONL-манифесты и изображения;
- модель: `Qwen/Qwen3.5-0.8B`;
- метод: LoRA fine-tuning, `r=16`, `alpha=32`, `dropout=0.05`;
- train/val: 38 019 / 1 981 примеров;
- лучший checkpoint: `checkpoint-4560`;
- итоговые training metrics: `runs/gqa_ru_qwen35_0_8b_lora_fast_v1/train_metrics.json`;
- сводка: `reports/benchmark_summary.json`;
- итоговый отчет: `reports/final_report.md`.

Важно: текущий репозиторий содержит обучение и loss-based validation. Полноценный evaluator для
GQA-ru/MMBench-ru accuracy в `scripts/eval.py` пока не реализован, поэтому `eval_loss` не следует
трактовать как финальный leaderboard-score.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Альтернатива через `uv`:

```bash
uv sync
```

## Формат JSONL для обучения

Каждая строка должна быть JSON-объектом как минимум с вопросом, ответом и путём к изображению.
Поддерживаются ключи изображения: `image`, `image_path`, `image_file`, `image_filename`.

```json
{"image": "000123.jpg", "question": "Что изображено на картинке?", "answer": "Красный автобус"}
```

Путь к изображению резолвится относительно `image_root` из `configs/data/*.yaml`.

## Подготовка DeepVK GQA-ru

Сначала можно скачать весь dataset repository локально:

```bash
python3 scripts/download_gqa_ru_snapshot.py \
  --output-dir data/raw/hf_gqa_ru_snapshot
```

После этого собрать локальный train/val/test из скачанного snapshot:

```bash
python3 scripts/prepare_gqa_ru_dataset.py \
  --output-root data/gqa_ru \
  --source-dir data/raw/hf_gqa_ru_snapshot \
  --val-size 0.05 \
  --answer-field answer
```

Скрипт читает parquet из локального snapshot, сохраняет изображения в
`data/gqa_ru/images/` и собирает:
- `data/gqa_ru/train.jsonl`
- `data/gqa_ru/val.jsonl`
- `data/gqa_ru/test.jsonl`

Для быстрого smoke-run можно ограничить объём:

```bash
python3 scripts/prepare_gqa_ru_dataset.py \
  --source-dir data/raw/hf_gqa_ru_snapshot \
  --train-max-images 20 \
  --test-max-images 10 \
  --train-max-samples 100 \
  --test-max-samples 50
```

## Запуск обучения

```bash
./scripts/train.sh --config configs/experiments/gqa_ru_qwen25vl_lora_v1.yaml
```

Полный пример:

```bash
./scripts/train.sh \
  --config configs/experiments/gqa_ru_qwen25vl_lora_v1.yaml \
  --device cuda:0 \
  --output-dir checkpoints/gqa_ru_qwen25vl_lora_v1
```

Скрипт загружает `Qwen/Qwen2.5-VL-3B-Instruct`, применяет LoRA из `configs/train/lora_ft.yaml`,
читает JSONL-манифесты и запускает `transformers.Trainer`.

Быстрый вариант для прогона и отладки:

```bash
python3 scripts/train.py \
  --config configs/experiments/gqa_ru_qwen35_0_8b_lora_fast_v1.yaml \
  --device cuda:0 \
  --output-dir checkpoints/gqa_ru_qwen35_0_8b_lora_fast_v1
```

Этот вариант использует меньшую модель, `max_seq_length: 1024`, `image_resolution: 336`,
LoRA `r: 16` и `1` эпоху.

## Проверка согласованности репозитория

```bash
python scripts/check_repo_consistency.py
```

Скрипт проверяет наличие обязательных папок/документов и корректность ссылок на конфиги в experiment YAML.
