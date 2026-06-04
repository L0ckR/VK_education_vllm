---
base_model: Qwen/Qwen3.5-0.8B
library_name: peft
pipeline_tag: image-text-to-text
tags:
- lora
- peft
- vk-education
- deepvk
- gqa-ru
- visual-question-answering
- lmms-eval
datasets:
- deepvk/GQA-ru
---

# vk-vlm-gqa-ru-qwen35-08b-lora

Мультимодальный LoRA-адаптер для `Qwen/Qwen3.5-0.8B`, обученный на открытом датасете
VK/DeepVK `GQA-ru` для русскоязычного visual question answering.

Автор: Ибрагимов Далгат Магомедалиевич, МАИ институт 8, группа М8О-308Б-32.

## Результат

Официальная оценка выполнена через `lmms-eval` на полном `gqa-ru` testdev split без `--limit`.
Модель получала изображения и вопросы. Для воспроизводимого короткого ответа использовался
`enable_thinking=False`.

| Модель | Samples | ExactMatch | Correct |
|---|---:|---:|---:|
| `Qwen/Qwen3.5-0.8B` | 12 216 | 0.2862 | 3 496 |
| LoRA adapter | 12 216 | 0.4832 | 5 903 |

Улучшение: **+0.1970 ExactMatch absolute**, **+68.85% relative**, **+2 407** правильных ответов.

## Данные

Использован открытый датасет `deepvk/GQA-ru`. Локальные JSONL-манифесты содержали путь к
изображению, русский вопрос и эталонный ответ.

| Split | Samples |
|---|---:|
| train | 38 019 |
| validation | 1 981 |
| testdev | 12 216 |

## Обучение

Обучение было мультимодальным: processor получал изображения, вопросы и ответы. Vision encoder
оставался замороженным, а LoRA обучалась в language model attention слоях, адаптируя обработку
visual tokens для русскоязычного VQA.

| Параметр | Значение |
|---|---|
| Base model | `Qwen/Qwen3.5-0.8B` |
| Adapter | LoRA |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Rank / alpha / dropout | `16 / 32 / 0.05` |
| Epochs | `1.0` |
| Batch size | `8` |
| Learning rate | `2e-4` |
| Precision | `bf16` |
| Seed | `42` |
| Best checkpoint | `checkpoint-4560` |

| Training metric | Value |
|---|---:|
| train_loss | 0.04432422036801592 |
| eval_loss | 0.4337001144886017 |
| train_runtime_sec | 6219.1947 |

## Использование

```python
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

base = "Qwen/Qwen3.5-0.8B"
adapter = "lockR/vk-vlm-gqa-ru-qwen35-08b-lora"

processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(base, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
```

## Ограничения

- Vision encoder не дообучался: LoRA применяется только к language model слоям.
- Результат измерен на GQA-ru; качество на других доменах и MMBench-ru не подтверждено.
- Модель может наследовать ограничения и смещения базовой модели и датасета.

## Репозиторий проекта

https://github.com/L0ckR/VK_education_vllm
