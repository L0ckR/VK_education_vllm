# Runbook: запуск на GPU-сервере

Ниже приведены базовые команды для запуска обучения, оценки и предсказания.

## 1) Подготовка окружения

```bash
cd /path/to/VK_education_vllm
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
# при необходимости: pip install -r requirements.txt
```

## 2) Обучение

```bash
./scripts/train.sh \
  --config configs/train.yaml \
  --device cuda:0 \
  --output-dir outputs/train_run_01
```

С возобновлением из чекпоинта:

```bash
./scripts/train.sh \
  --config configs/train.yaml \
  --device cuda:0 \
  --output-dir outputs/train_run_01 \
  --resume outputs/train_run_01/checkpoints/last.ckpt
```

## 3) Оценка

```bash
./scripts/eval.sh \
  --config configs/eval.yaml \
  --device cuda:0 \
  --output-dir outputs/eval_run_01 \
  --resume outputs/train_run_01/checkpoints/best.ckpt
```

## 4) Предсказание (inference)

```bash
./scripts/predict.sh \
  --config configs/predict.yaml \
  --device cuda:0 \
  --output-dir outputs/predict_run_01 \
  --resume outputs/train_run_01/checkpoints/best.ckpt
```

## Примечания

- Параметр `--config` обязателен для всех скриптов.
- `--device`, `--output-dir`, `--resume` — опциональны.
- `scripts/predict.sh` запускает `scripts/eval.py` в режиме `--mode predict`.
