# mlflow_example

Пример использования MLflow для трекинга экспериментов.

## Запуск

```bash
python3 runner.py
```

## Параметры с лучшим ROC-AUC (0.85)

- **process_data:** `train_size` — не задан (вся выборка)
- **train:** `model_type: xgboost`, `n_estimators: 50`, `max_depth: 6`

Текущие значения в `params/*.yaml` настроены на лучший результат. Отчёт — [REPORT.md](REPORT.md).
