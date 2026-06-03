# Подробное описание решения

## 1. Контекст

Проект выполняется на открытых данных VK (DeepVK VLM collection) с целью получить
воспроизводимый baseline на GQA-ru для русскоязычного question answering по изображениям.

## 2. Архитектура решения

- **Код (`src/`)**: утилиты, точка входа train/eval, функции воспроизводимости.
- **Конфиги (`configs/`)**: декомпозиция на data/model/train/eval/inference + experiments.
- **Скрипты (`scripts/`)**: единый CLI-контур запуска на GPU-машине.
- **Документация (`docs/`)**: постановка задачи, runbook, отчётный шаблон.
- **Исследования (`notebooks/eda/`)**: EDA и диагностический анализ ошибок.

## 3. Как используются данные VK

1. Берутся открытые датасеты из DeepVK VLM collection.
2. Формируются JSONL манифесты с путями к изображениям и полями вопрос/ответ.
3. Данные подключаются через `configs/data/*.yaml`.
4. Эксперименты запускаются через `configs/experiments/*.yaml`.

## 4. Процедура обучения

1. Выбирается experiment-конфиг (`gqa_ru_*` или `mmbench_ru_*`).
2. Подгружаются ссылки на model/data/train/eval/inference конфиги.
3. На GPU запускается train.
4. Метрики и артефакты сохраняются в `runs/` и `checkpoints/`.

Фактический обученный запуск:
- experiment: `gqa_ru_qwen35_0_8b_lora_fast_v1`;
- base model: `Qwen/Qwen3.5-0.8B`;
- данные: `deepvk/GQA-ru`, 38 019 train и 1 981 validation примеров;
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`;
- лучший checkpoint: `checkpoint-4560`;
- best validation loss: `0.4337001144886017`;
- HF-артефакт: https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen35-08b-lora.

### 4.1 Реальная оценка

Добавлен evaluator в `scripts/eval.py` / `src/vk_vlm_project/evaluate.py`.

Поддерживаемые режимы:
- teacher-forced answer loss/perplexity;
- генеративный exact match и token F1;
- сравнение исходной base model и LoRA-адаптера через `--compare-base`;
- сохранение `metrics.json` и `predictions.jsonl`.

Фактическое сравнение с оригинальной моделью:

| Прогон | Base | Adapter | Улучшение |
|---|---:|---:|---:|
| answer loss, val 200 | 5.169734188625889 | 2.53404495023912 | 50.98% |
| exact match, val 50 | 0.18 | 0.36 | +0.18 |
| token F1, val 50 | 0.20 | 0.36 | +0.16 |

## 5. Что сдаётся как результат

- обученная модель или LoRA-адаптер;
- таблица фактических метрик обучения и validation loss;
- заполненный отчёт (`reports/final_report.md`);
- (опционально) презентация с ключевыми выводами.

## 6. Критерии готовности

- цель, задачи и ожидаемые результаты зафиксированы;
- явно описано использование открытых данных VK;
- есть материалы по обученной модели;
- есть отдельный файл с подробным описанием решения.

Ограничение текущей версии: evaluator является текстовым QA-прокси без передачи изображения в
модель, потому что доступный checkpoint сохранен как `CAUSAL_LM` LoRA-адаптер. Результат является
реальным сравнением adapter vs original base model, но не финальным мультимодальным leaderboard
submission.
