# VK Education: Vision-Language Modeling проект

**HF artifact:** https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen25vl-3b-lora-smoke
**Best VLM run:** `gqa_ru_qwen25vl_lora_smoke_v1`, LoRA adapter for
`Qwen/Qwen2.5-VL-3B-Instruct`, `eval_loss=0.47339919209480286`.
**Official benchmark smoke:** `lmms-eval` task `gqa-ru`, `deepvk/GQA-ru`
`testdev_balanced_instructions`, `limit=100`: ExactMatch improved from `0.39` to `0.48`
(+0.09 absolute, +23.1% relative) versus the original VLM.

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

Опубликован основной LoRA-адаптер для настоящей VLM:
https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen25vl-3b-lora-smoke

Автор проекта: Ибрагимов Далгат Магомедалиевич, МАИ институт 8, группа М8О-308Б-32.

Фактический запуск:
- данные: `deepvk/GQA-ru`, подготовлены в локальные JSONL-манифесты и изображения;
- модель: `Qwen/Qwen2.5-VL-3B-Instruct`;
- метод: LoRA fine-tuning, `r=16`, `alpha=32`, `dropout=0.05`;
- train/val в smoke-обучении: 1 000 / 100 примеров из GQA-ru;
- итоговые training metrics: `runs/gqa_ru_qwen25vl_lora_smoke_v1/train_metrics.json`;
- официальный benchmark smoke-test:
  `runs/lmms_eval/gqa_ru_qwen25vl_lora_smoke_limit100/artifacts__merged_qwen25vl_gqa_ru_smoke/`;
- baseline на той же выборке:
  `runs/lmms_eval/gqa_ru_qwen25vl_base_limit100/Qwen__Qwen2.5-VL-3B-Instruct/`;
- сводка: `reports/benchmark_summary.json`;
- итоговый отчет: `reports/final_report.md`.

Важно: `lmms-eval --limit 100` используется как ограниченный официальный smoke-прогон, а не как
полный leaderboard score. При этом сравнение выполнено на одной и той же official task
конфигурации `gqa-ru` и показывает улучшение относительно исходной VLM.

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
  --config configs/experiments/gqa_ru_qwen25vl_lora_smoke_v1.yaml \
  --device cuda:0 \
  --output-dir checkpoints/gqa_ru_qwen25vl_lora_smoke_v1
```

Этот вариант использует настоящую VLM `Qwen/Qwen2.5-VL-3B-Instruct`, `max_seq_length: 1024`,
`image_resolution: 336`, LoRA `r: 16` и ограничение `max_train_samples: 1000`.

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

## Официальный benchmark smoke через lmms-eval

```bash
python -m lmms_eval eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks gqa-ru \
  --batch_size 1 \
  --limit 100 \
  --log_samples \
  --output_path runs/lmms_eval/gqa_ru_qwen25vl_base_limit100 \
  --device cuda:0 \
  --trust_remote_code
```

Для адаптера перед оценкой LoRA была смержена с базовой VLM в локальный ignored-артефакт
`artifacts/merged_qwen25vl_gqa_ru_smoke`, после чего запущена та же команда с
`--model_args pretrained=artifacts/merged_qwen25vl_gqa_ru_smoke`.
