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

Фактический основной обученный запуск:
- experiment: `gqa_ru_qwen35_0_8b_lora_fast_v1`;
- base model: `Qwen/Qwen3.5-0.8B`;
- данные: `deepvk/GQA-ru`, 38 019 train и 1 981 validation примеров;
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`;
- best validation loss: `0.4337001144886017`;
- HF-артефакт: https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen35-08b-lora.

Обучение было мультимодальным: data collator передавал processor изображения, вопросы и ответы.
Vision encoder оставался замороженным, а LoRA обучалась в language model attention projection
слоях. Это адаптирует обработку visual tokens без дорогого полного дообучения vision encoder.

### 4.1 Реальная оценка

Основная оценка выполнена официальным benchmark runner `lmms-eval` на задаче `gqa-ru`.

Параметры:
- task: `gqa-ru`;
- dataset: `deepvk/GQA-ru`;
- subset: `testdev_balanced_instructions`;
- split: `testdev`;
- metric: `exact_match`;
- effective samples: `12 216`;
- base: `Qwen/Qwen3.5-0.8B`;
- adapter: LoRA, локально смерженный с base model для совместимости с `lmms-eval`.

Фактическое сравнение с оригинальной моделью:

| Прогон | Base VLM | Adapter | Улучшение |
|---|---:|---:|---:|
| `lmms-eval gqa-ru`, full testdev | 0.2862 | 0.4832 | +0.1970 |

Оценка выполнена без `--limit` через мультимодальный backend `qwen3_5`. Бизнес-метрика для
проекта: `ExactMatch` на GQA-ru, то есть доля вопросов, где сгенерированный короткий ответ совпал
с эталонным после нормализации регистра и пунктуации.

Дополнительный полный эксперимент `Qwen/Qwen2.5-VL-3B-Instruct` выполняется тем же
мультимодальным training pipeline. Его итоговый full benchmark будет добавлен после завершения.

## 5. Что сдаётся как результат

- обученная VLM или LoRA-адаптер;
- таблица фактических метрик обучения и validation loss;
- заполненный отчёт (`reports/final_report.md`);
- (опционально) презентация с ключевыми выводами.

## 6. Критерии готовности

- цель, задачи и ожидаемые результаты зафиксированы;
- явно описано использование открытых данных VK;
- есть материалы по обученной модели;
- есть отдельный файл с подробным описанием решения.

Ограничение текущей версии: vision encoder заморожен, а MMBench-ru еще не измерен. Следующая
абляция должна проверить LoRA для multimodal projector/merger и последних vision encoder слоев.
