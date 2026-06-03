---
base_model: Qwen/Qwen3.5-0.8B
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- peft
- vk-education
- deepvk
- gqa-ru
- visual-question-answering
datasets:
- deepvk/GQA-ru
---

# vk-vlm-gqa-ru-qwen35-08b-lora

LoRA-адаптер, обученный для проекта VK Education Vision-Language Modeling на открытых данных
VK/DeepVK GQA-ru.

Автор: Ибрагимов Далгат Магомедалиевич, МАИ институт 8, группа М8О-308Б-32.

## Данные

Использован открытый датасет `deepvk/GQA-ru` из коллекции DeepVK VLM на Hugging Face. Данные были
приведены к JSONL-формату image/question/answer и использованы для обучения VQA-style модели.

Локальные размеры split в запуске:

| Split | Samples |
|---|---:|
| train | 38 019 |
| validation | 1 981 |
| test | 12 216 |

## Обучение

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

Лучший checkpoint: `checkpoint-4560`, выбран по `eval_loss`.

## Метрики

| Metric | Value |
|---|---:|
| train_loss | 0.04432422036801592 |
| eval_loss | 0.4337001144886017 |
| train_runtime_sec | 6219.1947 |
| train_samples_per_second | 6.113 |
| eval_samples_per_second | 17.075 |

Ограничение: сохраненная метрика является validation loss от `transformers.Trainer`. Полный
benchmark scoring для GQA-ru/MMBench-ru accuracy в текущем репозитории еще не реализован.

## Использование

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("lockR/vk-vlm-gqa-ru-qwen35-08b-lora", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "lockR/vk-vlm-gqa-ru-qwen35-08b-lora")
```

## Репозиторий проекта

https://github.com/L0ckR/VK_education_vllm
