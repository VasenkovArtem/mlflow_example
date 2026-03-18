# mlflow_example

Пример использования MLflow для трекинга экспериментов.

## Домашнее задание (базовый уровень)

- **Tracking URI:** `http://158.160.2.37:5000/`
- **Experiment:** `homework_Nikulin` (все run пайплайна попадают сюда)

### Один запуск пайплайна

В контейнере или в venv с зависимостями из `requirements.txt`:

```bash
python3 runner.py
```

Параметры шагов: `params/process_data.yaml`, `params/train.yaml`, `params/evaluate.yaml`.

### Серия экспериментов (10+ run)

```bash
python3 run_batch_experiments.py
```

Скрипт выполнит 15 запусков с разными конфигурациями (разрезы: размер train, тип модели, `C` у LR, набор признаков), затем вернёт базовые `params/*.yaml`.

### Лучший run по ROC-AUC

После прогона откройте MLflow UI, найдите run с максимальным `roc_auc`, при необходимости обновите `params/*_best.yaml` и опишите ссылку в [`report.md`](report.md).

### Отчёт

Шаблон отчёта с разрезами и местами под ссылки: [`report.md`](report.md).

## Сборка окружения (Docker)

```bash
docker compose up -d --build
```

Рабочая директория в контейнере: `/app` (репозиторий смонтирован в `/app`).
