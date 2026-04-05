# VK Education: Vision-Language Modeling проект

Репозиторий подготовлен под проект VK Education по обучению и оценке VLM-моделей на открытых
данных VK (коллекция DeepVK на Hugging Face), включая бенчмарки **GQA-ru** и **MMBench-ru**.

## Цель репозитория

Дать воспроизводимый каркас проекта, где:
- основной Python-код и утилиты находятся в `src/`;
- запуск выполняется через скрипты в `scripts/`;
- параметры экспериментов зафиксированы в `configs/`;
- описание проекта и артефактов оформлено в `docs/`;
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
- `data/` — структура для манифестов/примеров без тяжёлых артефактов.
- `notebooks/eda/` — место для исследовательских ноутбуков и EDA.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Быстрый запуск (без фактического обучения)

```bash
./scripts/train.sh --config configs/experiments/gqa_ru_qwen25vl_lora_v1.yaml
./scripts/eval.sh --config configs/experiments/gqa_ru_qwen25vl_lora_v1.yaml
./scripts/predict.sh --config configs/experiments/gqa_ru_qwen25vl_lora_v1.yaml
```

> Эти команды сейчас валидируют аргументы и конфиги, но не запускают тяжёлое обучение. Реальный запуск
> предполагается на машине с GPU.

## Проверка согласованности репозитория

```bash
python scripts/check_repo_consistency.py
```

Скрипт проверяет наличие обязательных папок/документов и корректность ссылок на конфиги в experiment YAML.
