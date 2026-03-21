# mlflow_example — ДЗ по MLflow (базовый уровень)

**Experiment в MLflow:** `homework_Nikulin`  
**Tracking URI:** http://158.160.2.37:5000/

## Структура репозитория (для проверки)

| Что смотреть | Где |
|--------------|-----|
| Код пайплайна | `scripts/process_data.py`, `scripts/train.py`, `scripts/evaluate.py`, `runner.py` |
| Параметры шагов | `params/process_data.yaml`, `params/train.yaml`, `params/evaluate.yaml` |
| **Лучший run по ROC-AUC** (воспроизведение) | `params/process_data_best.yaml`, `params/train_best.yaml`, `params/evaluate_best.yaml` |
| **Отчёт** (разрезы, таблицы, выводы, ссылки на run) | [`report.md`](report.md) |

## Запуск

Зависимости: `pip install -r requirements.txt` (или Docker, см. ниже).

**Один прогон пайплайна** (текущие `params/*.yaml`):

```bash
python3 runner.py
```

**Воспроизвести лучший по ROC-AUC run** (gradient boosting, см. `report.md`):

```bash
cp params/process_data_best.yaml params/process_data.yaml
cp params/train_best.yaml params/train.yaml
cp params/evaluate_best.yaml params/evaluate.yaml
python3 runner.py
```

**Серия из 15 экспериментов** (разрезы: `train_size`, тип модели, `C` у LR, набор признаков):

```bash
# в контейнере, из /app:
python3 run_batch_experiments.py
```

## Docker

```bash
docker compose up -d --build
docker exec -it <имя_контейнера> python3 /app/runner.py
```

Репозиторий монтируется в `/app`.

## Продвинутый уровень (часть II)

Отчёт по экспериментам коллеги (пример: `homework_Bazhenov`): [`results_Bazhenov.md`](results_Bazhenov.md) — загрузить на шаг Степика (имя файла уточнить у преподавателя: `results_<фамилия>.md`).

## Продвинутый уровень (часть I)

После шага `process_data` в каждый run (при вызове через `runner.py`) в MLflow в артефакты попадает обучающая выборка:

- `train_dataset/X_train.csv`
- `train_dataset/y_train.csv`

Чтобы выполнить требование «не менее трёх запусков» — достаточно три раза выполнить `python3 runner.py` с разными `params` (или прогнать `run_batch_experiments.py`: артефакты будут у всех run).

## Исходный репозиторий курса

Форк от [VasenkovArtem/mlflow_example](https://github.com/VasenkovArtem/mlflow_example).
