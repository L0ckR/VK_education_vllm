# VK Education: Vision-Language Modeling проект

**HF artifact:** https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen35-08b-lora
**Best completed VLM run:** `gqa_ru_qwen35_0_8b_lora_fast_v1`, LoRA adapter for
`Qwen/Qwen3.5-0.8B`, `eval_loss=0.4337001144886017`.
**Full official benchmark:** `lmms-eval` task `gqa-ru`, all `12 216` testdev samples:
ExactMatch improved from `0.2862` to `0.4832` (+0.1970 absolute, +68.85% relative).

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

Опубликован основной LoRA-адаптер:
https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen35-08b-lora

Автор проекта: Ибрагимов Далгат Магомедалиевич, МАИ институт 8, группа М8О-308Б-32.

Фактический запуск:
- данные: `deepvk/GQA-ru`, подготовлены в локальные JSONL-манифесты и изображения;
- модель: `Qwen/Qwen3.5-0.8B`;
- метод: LoRA fine-tuning, `r=16`, `alpha=32`, `dropout=0.05`;
- train/val: 38 019 / 1 981 примеров из GQA-ru;
- обучение мультимодальное: processor получает изображения, вопросы и ответы;
- vision encoder заморожен, LoRA обучается в language model attention слоях;
- итоговые training metrics: `runs/gqa_ru_qwen35_0_8b_lora_fast_v1/train_metrics.json`;
- полный официальный benchmark:
  `runs/lmms_eval/gqa_ru_qwen35_lora_full/artifacts__merged_qwen35_gqa_ru_full/`;
- baseline на полном testdev:
  `runs/lmms_eval/gqa_ru_qwen35_base_full/Qwen__Qwen3.5-0.8B/`;
- сводка: `reports/benchmark_summary.json`;
- итоговый отчет: `reports/final_report.md`.

Дополнительный full-эксперимент с `Qwen/Qwen2.5-VL-3B-Instruct` выполняется через
`configs/experiments/gqa_ru_qwen25vl_lora_full_v1.yaml`. Его результат не включен в основную
таблицу до завершения обучения и полного base-vs-adapter benchmark.

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
./scripts/train.sh --config configs/experiments/gqa_ru_qwen35_0_8b_lora_fast_v1.yaml
```

Полный пример:

```bash
./scripts/train.sh \
  --config configs/experiments/gqa_ru_qwen35_0_8b_lora_fast_v1.yaml \
  --device cuda:0 \
  --output-dir checkpoints/gqa_ru_qwen35_0_8b_lora_fast_v1
```

Скрипт загружает мультимодальную `Qwen/Qwen3.5-0.8B`, применяет LoRA из
`configs/train/lora_ft_fast.yaml`, читает JSONL-манифесты с изображениями и запускает
`transformers.Trainer`.

Второй full-эксперимент для Qwen2.5-VL:

```bash
python3 scripts/train.py \
  --config configs/experiments/gqa_ru_qwen25vl_lora_full_v1.yaml \
  --device cuda:0 \
  --output-dir checkpoints/gqa_ru_qwen25vl_lora_full_v1
```

Оба запуска передают изображения в модель. Vision encoder остается замороженным, а LoRA обучается
в language model attention слоях.

## Проверка согласованности репозитория

```bash
python scripts/check_repo_consistency.py
```

Скрипт проверяет наличие обязательных папок/документов и корректность ссылок на конфиги в experiment YAML.

## Запуск реальной оценки

```bash
python3 scripts/eval.py \
  --config configs/experiments/gqa_ru_qwen35_0_8b_lora_fast_v1.yaml \
  --resume checkpoints/gqa_ru_qwen35_0_8b_lora_fast_v1 \
  --split val \
  --max-samples 200 \
  --compare-base \
  --no-generate
```

Для генеративной smoke-оценки с prediction JSONL уберите `--no-generate` и задайте меньший
`--max-samples`, например `50`.

## Полный официальный benchmark через lmms-eval

```bash
./scripts/run_full_gqa_ru_pipeline.sh
```

Скрипт запускает `lmms-eval gqa-ru` без `--limit`, сохраняет result JSON и sample logs для base и
adapter, а также содержит очередь полного Qwen2.5-VL эксперимента.
