# Итоговый отчет

## 1. Общая информация

- Название проекта: дообучение VLM/QA-модели на открытых данных VK DeepVK.
- Автор: Ибрагимов Далгат Магомедалиевич.
- Организация: МАИ, институт 8, группа М8О-308Б-32.
- Репозиторий: https://github.com/L0ckR/VK_education_vllm
- HF-артефакт: https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen35-08b-lora
- Версия отчета: 2026-06-03.

## 2. Цель и задачи

Цель проекта: получить воспроизводимый baseline для русскоязычного visual question answering на
открытых данных VK/DeepVK и подготовить обученный LoRA-адаптер как внешний артефакт.

Задачи:
- изучить открытые VLM-датасеты VK/DeepVK;
- подготовить структуру проекта, конфиги, скрипты обучения и документацию;
- сформировать локальные JSONL-манифесты GQA-ru;
- обучить LoRA-адаптер;
- сохранить метрики, описание решения и ссылку на модельный артефакт.

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
| Experiment | `gqa_ru_qwen35_0_8b_lora_fast_v1` |
| Base model | `Qwen/Qwen3.5-0.8B` |
| Adapter | LoRA |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LoRA rank / alpha / dropout | `16 / 32 / 0.05` |
| Epochs | `1.0` |
| Batch size | `8` |
| LR | `2e-4` |
| Precision | `bf16` |
| Seed | `42` |

Лучший checkpoint по validation loss: `checkpoint-4560`.

## 5. Результаты

| Метрика | Значение |
|---|---:|
| train_loss | 0.04432422036801592 |
| eval_loss | 0.4337001144886017 |
| train_runtime_sec | 6219.1947 |
| train_samples_per_second | 6.113 |
| train_steps_per_second | 0.764 |
| eval_samples_per_second | 17.075 |

Важно: `eval_loss` является loss-based метрикой Trainer на validation split. Это не полноценная
accuracy-оценка GQA-ru и не leaderboard-метрика MMBench-ru. Для финального сравнения моделей нужно
добавить генеративный evaluator, сохранить predictions и посчитать exact match/accuracy.

## 6. Артефакты

- LoRA-адаптер на Hugging Face: https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen35-08b-lora
- Метрики запуска: `runs/gqa_ru_qwen35_0_8b_lora_fast_v1/train_metrics.json`
- Сводка метрик: `reports/benchmark_summary.json`
- Подробное описание решения: `docs/solution_detailed.md`
- Описание проекта: `docs/project_description.md`

## 7. Ограничения и дальнейшие шаги

Текущий результат закрывает проектное требование по описанию проекта, использованию открытых
данных VK и публикации обученного LoRA-адаптера. Главный технический риск: в репозитории пока нет
полного evaluator для GQA-ru/MMBench-ru accuracy, поэтому качество модели подтверждено только
validation loss, а не целевой соревновательной метрикой.

Следующие шаги:
- реализовать `scripts/eval.py` для генерации ответов и расчета exact match/accuracy;
- прогнать GQA-ru test и MMBench-ru;
- добавить `reports/predictions_*.jsonl` и итоговую таблицу benchmark-score;
- сравнить текущий LoRA-adapter с более сильной VLM-базой `Qwen/Qwen2.5-VL-3B-Instruct`.
