# Итоговый отчет

## 1. Общая информация

- Название проекта: дообучение VLM/QA-модели на открытых данных VK DeepVK.
- Автор: Ибрагимов Далгат Магомедалиевич.
- Организация: МАИ, институт 8, группа М8О-308Б-32.
- Репозиторий: https://github.com/L0ckR/VK_education_vllm
- HF-артефакт: https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen25vl-3b-lora-smoke
- Версия отчета: 2026-06-04.

## 2. Цель и задачи

Цель проекта: получить воспроизводимый baseline для русскоязычного visual question answering на
открытых данных VK/DeepVK, обучить LoRA-адаптер для настоящей VLM и подтвердить улучшение
официальной benchmark-метрикой.

Задачи:
- изучить открытые VLM-датасеты VK/DeepVK;
- подготовить структуру проекта, конфиги, скрипты обучения и документацию;
- сформировать локальные JSONL-манифесты GQA-ru;
- обучить LoRA-адаптер для `Qwen/Qwen2.5-VL-3B-Instruct`;
- прогнать официальный `lmms-eval` benchmark `gqa-ru`;
- сохранить метрики, sample-логи, описание решения и ссылку на модельный артефакт.

## 3. Данные VK

Основной использованный датасет: `deepvk/GQA-ru`.

Данные применялись как задача VQA: каждая запись содержит изображение, вопрос на русском языке и
эталонный ответ. Локальная подготовка сформировала:

| Split | Количество примеров |
|---|---:|
| train | 38 019 |
| validation | 1 981 |
| test | 12 216 |

MMBench-ru добавлен в конфигурации проекта как следующий benchmark для сравнения, но в доступных
артефактах текущего запуска нет сохраненного MMBench-ru score.

## 4. Модель и обучение

| Параметр | Значение |
|---|---|
| Experiment | `gqa_ru_qwen25vl_lora_smoke_v1` |
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Adapter | LoRA |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LoRA rank / alpha / dropout | `16 / 32 / 0.05` |
| Epochs | `1.0` |
| Batch size | `1`, gradient accumulation `16` |
| LR | `2e-4` |
| Precision | `bf16` |
| Seed | `42` |
| Train / eval samples | `1 000 / 100` |

Финальный LoRA-адаптер сохранен в `checkpoints/gqa_ru_qwen25vl_lora_smoke_v1` и опубликован на
Hugging Face. Для официальной оценки адаптер был локально смержен с базовой моделью в ignored
директорию `artifacts/merged_qwen25vl_gqa_ru_smoke`.

## 5. Результаты

### 5.1 Метрики обучения

| Метрика | Значение |
|---|---:|
| train_loss | 0.6548236324673608 |
| eval_loss | 0.47339919209480286 |
| train_runtime_sec | 714.4626 |
| train_samples_per_second | 1.4 |
| train_steps_per_second | 0.088 |
| eval_samples_per_second | 7.061 |

### 5.2 Официальная benchmark-метрика

Проверка выполнена через официальный пакет `lmms-eval` на задаче `gqa-ru`:

- dataset: `deepvk/GQA-ru`;
- subset: `testdev_balanced_instructions`;
- split: `testdev`;
- metric: `exact_match`, case/punctuation insensitive;
- effective samples: `100`;
- prompt suffix: `Ответь одним словом.`

`lmms-eval` предупреждает, что `--limit` предназначен для тестирования, поэтому результат ниже
зафиксирован как официальный benchmark smoke, а не полный leaderboard score.

| Модель | ExactMatch | stderr | Correct / 100 |
|---|---:|---:|
| `Qwen/Qwen2.5-VL-3B-Instruct` | 0.39 | 0.04902071300001975 | 39 |
| LoRA adapter | 0.48 | 0.050211673156867795 | 48 |

Улучшение: **+0.09 ExactMatch absolute**, или **+23.1% relative** к исходной VLM.

Дополнительно на 20-sample smoke та же official task дала `0.55 -> 0.60`.

### 5.3 Предыдущий текстовый QA-прокси

В репозитории также сохранен предыдущий `CAUSAL_LM` LoRA-эксперимент
`gqa_ru_qwen35_0_8b_lora_fast_v1` как дополнительный baseline. Он улучшал loss-based QA proxy на
GQA-ru validation, но не являлся полноценной VLM-оценкой, поэтому не выбран основным результатом.

## 6. Артефакты

- LoRA-адаптер на Hugging Face: https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen25vl-3b-lora-smoke
- Метрики обучения VLM: `runs/gqa_ru_qwen25vl_lora_smoke_v1/train_metrics.json`
- Official baseline result: `runs/lmms_eval/gqa_ru_qwen25vl_base_limit100/Qwen__Qwen2.5-VL-3B-Instruct/20260604_034100_results.json`
- Official adapter result: `runs/lmms_eval/gqa_ru_qwen25vl_lora_smoke_limit100/artifacts__merged_qwen25vl_gqa_ru_smoke/20260604_034349_results.json`
- Official sample logs: `runs/lmms_eval/gqa_ru_qwen25vl_*_limit100/**/20260604_*_samples_gqa-ru.jsonl`
- Сводка метрик: `reports/benchmark_summary.json`
- Подробное описание решения: `docs/solution_detailed.md`
- Описание проекта: `docs/project_description.md`

## 7. Ограничения и дальнейшие шаги

Текущий результат закрывает проектное требование по описанию проекта, использованию открытых
данных VK, публикации обученного LoRA-адаптера и сравнению с оригинальной моделью по benchmark
метрике. Главный технический риск: официальный `lmms-eval` запуск ограничен `limit=100` из-за
времени, поэтому это не полный leaderboard score.

Следующие шаги:
- прогнать GQA-ru testdev без `--limit`;
- прогнать MMBench-ru;
- добавить `reports/predictions_*.jsonl` и итоговую таблицу benchmark-score;
- расширить обучение за пределы smoke-режима: больше train samples, больше эпох и подбор LoRA rank.
