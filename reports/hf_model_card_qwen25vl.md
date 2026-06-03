---
base_model: Qwen/Qwen2.5-VL-3B-Instruct
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

# vk-vlm-gqa-ru-qwen25vl-3b-lora-smoke

LoRA-адаптер для `Qwen/Qwen2.5-VL-3B-Instruct`, обученный для проекта VK Education
Vision-Language Modeling на открытых данных VK/DeepVK GQA-ru.

Автор: Ибрагимов Далгат Магомедалиевич, МАИ институт 8, группа М8О-308Б-32.

## Данные

Использован открытый датасет `deepvk/GQA-ru` из коллекции DeepVK VLM на Hugging Face. Данные
использовались как VQA: изображение, вопрос на русском языке и короткий эталонный ответ.

В smoke-обучении использовано 1 000 train и 100 validation примеров из подготовленных локальных
манифестов GQA-ru.

## Обучение

| Параметр | Значение |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Adapter | LoRA |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Rank / alpha / dropout | `16 / 32 / 0.05` |
| Epochs | `1.0` |
| Batch size / grad accumulation | `1 / 16` |
| Learning rate | `2e-4` |
| Precision | `bf16` |
| Seed | `42` |

Training metrics:

| Metric | Value |
|---|---:|
| train_loss | 0.6548236324673608 |
| eval_loss | 0.47339919209480286 |
| train_runtime_sec | 714.4626 |
| train_samples_per_second | 1.4 |
| eval_samples_per_second | 7.061 |

## Benchmark

Официальный benchmark smoke выполнен через `lmms-eval` на задаче `gqa-ru`:

| Модель | Samples | ExactMatch |
|---|---:|---:|
| `Qwen/Qwen2.5-VL-3B-Instruct` | 100 | 0.39 |
| LoRA adapter | 100 | 0.48 |

Улучшение: +0.09 ExactMatch absolute, +23.1% relative.

Ограничение: использован `lmms-eval --limit 100`, поэтому это bounded official-task smoke, а не
полный leaderboard score.

## Использование

```python
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

base = "Qwen/Qwen2.5-VL-3B-Instruct"
adapter = "lockR/vk-vlm-gqa-ru-qwen25vl-3b-lora-smoke"

processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
```

## Репозиторий проекта

https://github.com/L0ckR/VK_education_vllm
