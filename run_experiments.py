#!/usr/bin/env python3
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts import process_data, train, evaluate
import mlflow

from constants import MLFLOW_TRACKING_URI, EXPERIMENT_NAME

PARAMS_DIR = os.path.join(os.path.dirname(__file__), 'params')


def save_params(stage: str, params: dict):
    path = os.path.join(PARAMS_DIR, f'{stage}.yaml')
    with open(path, 'w') as f:
        yaml.dump({'params': params}, f, default_flow_style=False, allow_unicode=True)


def run_experiment(process_params, train_params, evaluate_params):
    save_params('process_data', process_params)
    save_params('train', train_params)
    save_params('evaluate', evaluate_params)

    with mlflow.start_run():
        process_data()
        train()
        evaluate()


FEATURES_BASE = ['age', 'education', 'capital.gain', 'hours.per.week', 'race', 'sex', 'occupation']
FEATURES_EXTENDED = FEATURES_BASE + ['workclass', 'marital.status', 'relationship', 'native.country', 'capital.loss']
FEATURES_MINIMAL = ['age', 'education', 'capital.gain', 'race', 'sex']


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    experiments = []

    # Разрез 1: Размер датасета (4 эксперимента)
    for train_size in [2000, 5000, 10000, 15000]:
        experiments.append({
            'process_data': {
                'features': FEATURES_BASE,
                'train_size': train_size,
            },
            'train': {
                'model_type': 'logistic_regression',
                'penalty': 'l2',
                'C': 0.9,
                'solver': 'lbfgs',
                'max_iter': 1000,
            },
            'evaluate': {'artifact_type': 'classification_report'},
        })

    for model_type, model_params in [
        ('logistic_regression', {'penalty': 'l2', 'C': 0.9, 'solver': 'lbfgs', 'max_iter': 1000}),
        ('decision_tree', {'max_depth': 10, 'min_samples_split': 5}),
        ('random_forest', {'n_estimators': 100, 'max_depth': 10}),
        ('gradient_boosting', {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}),
    ]:
        experiments.append({
            'process_data': {'features': FEATURES_BASE, 'train_size': 10000},
            'train': {'model_type': model_type, **model_params},
            'evaluate': {'artifact_type': 'confusion_matrix'},
        })

    for C, solver in [(0.1, 'lbfgs'), (1.0, 'lbfgs'), (10.0, 'saga')]:
        experiments.append({
            'process_data': {'features': FEATURES_BASE, 'train_size': 10000},
            'train': {
                'model_type': 'logistic_regression',
                'penalty': 'l2',
                'C': C,
                'solver': solver,
                'max_iter': 1000,
            },
            'evaluate': {'artifact_type': 'pr_curve'},
        })

    for features_set, features in [
        ('minimal', FEATURES_MINIMAL),
        ('base', FEATURES_BASE),
        ('extended', FEATURES_EXTENDED),
    ]:
        experiments.append({
            'process_data': {'features': features, 'train_size': 10000, 'features_set': features_set},
            'train': {
                'model_type': 'logistic_regression',
                'penalty': 'l2',
                'C': 0.9,
                'solver': 'lbfgs',
                'max_iter': 1000,
            },
            'evaluate': {'artifact_type': 'classification_report'},
        })

    print(f'Запуск {len(experiments)} экспериментов...')
    for i, exp in enumerate(experiments):
        print(f'\n--- Эксперимент {i + 1}/{len(experiments)} ---')
        run_experiment(exp['process_data'], exp['train'], exp['evaluate'])

    print(f'\nЗавершено! Всего запущено {len(experiments)} экспериментов.')
    print(f'Результаты: {MLFLOW_TRACKING_URI}')


if __name__ == '__main__':
    main()
