#!/usr/bin/env python3
"""
Серия экспериментов для ДЗ: 13 запусков, 3 разреза (размер данных, тип модели, C у LR).
Запуск: из корня репозитория, внутри окружения с зависимостями:
  python run_batch_experiments.py

Требуется доступ к MLflow: http://158.160.2.37:5000/
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PARAMS = ROOT / 'params'


def write_process(features: list, train_size: int) -> None:
    data = {
        'params': {
            'features': features,
            'train_size': train_size,
        }
    }
    with open(PARAMS / 'process_data.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def write_train(cfg: dict) -> None:
    data = {'params': cfg}
    with open(PARAMS / 'train.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def run_pipeline(run_name: str, slice_tag: str) -> None:
    env = {**os.environ, 'MLFLOW_RUN_NAME': run_name, 'MLFLOW_EXPERIMENT_SLICE': slice_tag}
    r = subprocess.run(
        [sys.executable, str(ROOT / 'runner.py')],
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    if r.returncode != 0:
        raise SystemExit(f'Ошибка в run {run_name}, код {r.returncode}')


FEATURES_FULL = [
    'race',
    'sex',
    'native.country',
    'occupation',
    'education',
    'capital.gain',
    'age',
    'hours.per.week',
]
FEATURES_MIN = ['education', 'occupation', 'capital.gain', 'age']


def main() -> None:
    os.chdir(ROOT)

    train_lr = {
        'model_type': 'logistic_regression',
        'logistic_regression': {
            'penalty': 'l2',
            'C': 0.9,
            'solver': 'lbfgs',
            'max_iter': 1000,
        },
        'decision_tree': {'max_depth': 8, 'min_samples_split': 4},
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 12,
            'min_samples_split': 2,
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
        },
        'xgboost': {
            'n_estimators': 80,
            'max_depth': 5,
            'learning_rate': 0.1,
        },
    }

    # Разрез 1: только train_size (модель LR, фичи базовые)
    for ts in (2000, 4000, 6000, 8000, 10000):
        write_process(FEATURES_FULL[:6], ts)
        write_train(train_lr)
        run_pipeline(f'slice_train_size_{ts}', 'train_size')

    # Разрез 2: только model_type (train_size=8000, те же 6 фичей)
    write_process(FEATURES_FULL[:6], 8000)
    for mt, extra in [
        ('logistic_regression', {}),
        ('decision_tree', {}),
        ('random_forest', {}),
        ('gradient_boosting', {}),
    ]:
        cfg = {**train_lr, 'model_type': mt}
        write_train(cfg)
        run_pipeline(f'slice_model_{mt}', 'model_type')

    # Разрез 3: только C у LR (train_size=8000)
    write_process(FEATURES_FULL[:6], 8000)
    for c_val in (0.01, 0.1, 1.0, 10.0):
        cfg = {
            **train_lr,
            'model_type': 'logistic_regression',
            'logistic_regression': {
                'penalty': 'l2',
                'C': c_val,
                'solver': 'lbfgs',
                'max_iter': 2000,
            },
        }
        write_train(cfg)
        run_pipeline(f'slice_lr_C_{c_val}', 'lr_hyperparams')

    # Бонус: разрез features (3 варианта)
    write_process(FEATURES_MIN, 8000)
    write_train(train_lr)
    run_pipeline('slice_features_minimal', 'features')

    write_process(FEATURES_FULL, 8000)
    write_train(train_lr)
    run_pipeline('slice_features_full', 'features')

    write_process(FEATURES_FULL[:6], 8000)
    write_train(train_lr)
    print('Готово: 15 запусков. params/process_data.yaml и params/train.yaml возвращены к базовым значениям.')


if __name__ == '__main__':
    main()
