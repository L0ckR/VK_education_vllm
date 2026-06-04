# Итоговый отчет

## 1. Общая информация

- Название проекта: дообучение VLM/QA-модели на открытых данных VK DeepVK.
- Автор: Ибрагимов Далгат Магомедалиевич.
- Организация: МАИ, институт 8, группа М8О-308Б-32.
- Репозиторий: https://github.com/L0ckR/VK_education_vllm
- HF-артефакт: https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen35-08b-lora
- Версия отчета: 2026-06-04.

## 2. Цель и задачи

Цель проекта: получить воспроизводимый baseline для русскоязычного visual question answering на
открытых данных VK/DeepVK, обучить LoRA-адаптер для настоящей VLM и подтвердить улучшение
официальной benchmark-метрикой.

Задачи:
- изучить открытые VLM-датасеты VK/DeepVK;
- подготовить структуру проекта, конфиги, скрипты обучения и документацию;
- сформировать локальные JSONL-манифесты GQA-ru;
- обучить LoRA-адаптер для мультимодальной `Qwen/Qwen3.5-0.8B`;
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
| Train / validation samples | `38 019 / 1 981` |

Обучение было мультимодальным: модель получала изображения, вопросы и ответы. Vision encoder
оставался замороженным, а LoRA обучалась в language model attention слоях. Лучший checkpoint по
`eval_loss`: `checkpoint-4560`. Для official evaluation адаптер был локально смержен с base model.

## 5. Результаты

### 5.1 Метрики обучения

| Метрика | Значение |
|---|---:|
| train_loss | 0.04432422036801592 |
| eval_loss | 0.4337001144886017 |
| train_runtime_sec | 6219.1947 |
| train_samples_per_second | 6.113 |
| train_steps_per_second | 0.764 |
| eval_samples_per_second | 17.075 |

### 5.2 Официальная benchmark-метрика

Проверка выполнена через официальный пакет `lmms-eval` на задаче `gqa-ru`:

- dataset: `deepvk/GQA-ru`;
- subset: `testdev_balanced_instructions`;
- split: `testdev`;
- metric: `exact_match`, case/punctuation insensitive;
- effective samples: `12 216`;
- prompt suffix: `Ответь одним словом.`

Оценка выполнена без `--limit` через мультимодальный backend `qwen3_5`; для короткого ответа
использован `enable_thinking=False`.

| Модель | ExactMatch | stderr | Correct / 12 216 |
|---|---:|---:|---:|
| `Qwen/Qwen3.5-0.8B` | 0.2861820563195809 | 0.004089480999753636 | 3 496 |
| LoRA adapter | 0.48321872953503603 | 0.004521458266039995 | 5 903 |

Улучшение: **+0.1970 ExactMatch absolute**, **+68.85% relative**, **+2 407** правильных ответов.

### 5.3 Второй эксперимент

Полный эксперимент `Qwen/Qwen2.5-VL-3B-Instruct` сейчас выполняется. До его завершения в отчете
сохранен только smoke-результат `0.39 -> 0.48` на 100 примерах, который не сравнивается напрямую с
полным Qwen3.5 testdev результатом.

## 6. Артефакты

- LoRA-адаптер на Hugging Face: https://huggingface.co/lockR/vk-vlm-gqa-ru-qwen35-08b-lora
- Метрики обучения VLM: `runs/gqa_ru_qwen35_0_8b_lora_fast_v1/train_metrics.json`
- Official baseline result: `runs/lmms_eval/gqa_ru_qwen35_base_full/Qwen__Qwen3.5-0.8B/20260604_141134_results.json`
- Official adapter result: `runs/lmms_eval/gqa_ru_qwen35_lora_full/artifacts__merged_qwen35_gqa_ru_full/20260604_145224_results.json`
- Сводка метрик: `reports/benchmark_summary.json`
- Подробное описание решения: `docs/solution_detailed.md`
- Описание проекта: `docs/project_description.md`

## 7. Ограничения и дальнейшие шаги

Текущий результат закрывает проектное требование по описанию проекта, использованию открытых
данных VK, публикации обученного LoRA-адаптера и сравнению с оригинальной моделью по benchmark
метрике на полном testdev split. Главный технический риск: vision encoder оставался замороженным,
поэтому адаптация выполнялась только через language model LoRA.

Следующие шаги:
- завершить полный Qwen2.5-VL эксперимент и сравнить две VLM;
- прогнать MMBench-ru;
- провести абляцию LoRA для projector/merger и последних vision encoder слоев;
- добавить анализ ошибок по типам вопросов.
