import os
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml

PARAMS_DIR = Path('/app/params')

PROCESS_DATA_PATH = PARAMS_DIR / 'process_data.yaml'
TRAIN_PATH = PARAMS_DIR / 'train.yaml'
EVALUATE_PATH = PARAMS_DIR / 'evaluate.yaml'


def read_yaml(path: Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


FEATURE_SET_A = [
    'age',
    'education.num',
    'hours.per.week',
]

FEATURE_SET_B = [
    'age',
    'education.num',
    'hours.per.week',
    'capital.gain',
    'capital.loss',
]

FEATURE_SET_C = [
    'age',
    'education.num',
    'hours.per.week',
    'capital.gain',
    'capital.loss',
    'occupation',
    'marital.status',
    'relationship',
    'sex',
]

EVALUATE_CONFIG = {
    'params': {
        'metrics': [
            'accuracy',
            'precision',
            'recall',
            'f1',
            'roc_auc',
            'pr_auc',
        ],
        'threshold': 0.5,
    }
}

LR_CONFIG = {
    'params': {
        'model_type': 'logistic_regression',
        'penalty': 'l2',
        'C': 1.0,
        'solver': 'lbfgs',
        'max_iter': 1000,
    }
}

DT_CONFIG = {
    'params': {
        'model_type': 'decision_tree',
        'max_depth': 8,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
    }
}

RF_CONFIG = {
    'params': {
        'model_type': 'random_forest',
        'n_estimators': 300,
        'max_depth': 10,
        'min_samples_split': 20,
        'min_samples_leaf': 5,
        'n_jobs': -1,
    }
}

XGB_CONFIG = {
    'params': {
        'model_type': 'xgboost',
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'tree_method': 'hist',
        'n_jobs': -1,
    }
}

EXPERIMENTS = [
    # Разрез 1: размер train_size
    {
        'name': 'slice1_train_size_1000',
        'tags': {
            'study_slice': 'train_size',
            'study_value': '1000',
            'study_hypothesis': 'Увеличение train_size повышает качество модели',
        },
        'process_data': {
            'params': {
                'features': FEATURE_SET_C,
                'train_size': 1000,
            }
        },
        'train': deepcopy(LR_CONFIG),
        'evaluate': deepcopy(EVALUATE_CONFIG),
    },
    {
        'name': 'slice1_train_size_3000',
        'tags': {
            'study_slice': 'train_size',
            'study_value': '3000',
            'study_hypothesis': 'Увеличение train_size повышает качество модели',
        },
        'process_data': {
            'params': {
                'features': FEATURE_SET_C,
                'train_size': 3000,
            }
        },
        'train': deepcopy(LR_CONFIG),
        'evaluate': deepcopy(EVALUATE_CONFIG),
    },
    {
        'name': 'slice1_train_size_5000',
        'tags': {
            'study_slice': 'train_size',
            'study_value': '5000',
            'study_hypothesis': 'Увеличение train_size повышает качество модели',
        },
        'process_data': {
            'params': {
                'features': FEATURE_SET_C,
                'train_size': 5000,
            }
        },
        'train': deepcopy(LR_CONFIG),
        'evaluate': deepcopy(EVALUATE_CONFIG),
    },
    {
        'name': 'slice1_train_size_10000',
        'tags': {
            'study_slice': 'train_size',
            'study_value': '10000',
            'study_hypothesis': 'Увеличение train_size повышает качество модели',
        },
        'process_data': {
            'params': {
                'features': FEATURE_SET_C,
                'train_size': 10000,
            }
        },
        'train': deepcopy(LR_CONFIG),
        'evaluate': deepcopy(EVALUATE_CONFIG),
    },

    # Разрез 2: тип модели
    {
        'name': 'slice2_model_logistic_regression',
        'tags': {
            'study_slice': 'model_type',
            'study_value': 'logistic_regression',
            'study_hypothesis': 'Ансамблевые модели дают ROC-AUC выше логистической регрессии',
        },
        'process_data': {
            'params': {
                'features': FEATURE_SET_C,
                'train_size': 10000,
            }
        },
        'train': deepcopy(LR_CONFIG),
        'evaluate': deepcopy(EVALUATE_CONFIG),
    },
    {
        'name': 'slice2_model_decision_tree',
        'tags': {
            'study_slice': 'model_type',
            'study_value': 'decision_tree',
            'study_hypothesis': 'Ансамблевые модели дают ROC-AUC выше логистической регрессии',
        },
        'process_data': {
            'params': {
                'features': FEATURE_SET_C,
                'train_size': 10000,
            }
        },
        'train': deepcopy(DT_CONFIG),
        'evaluate': deepcopy(EVALUATE_CONFIG),
    },
    {
        'name': 'slice2_model_random_forest',
        'tags': {
            'study_slice': 'model_type',
            'study_value': 'random_forest',
            'study_hypothesis': 'Ансамблевые модели дают ROC-AUC выше логистической регрессии',
        },
        'process_data': {
            'params': {
                'features': FEATURE_SET_C,
                'train_size': 10000,
            }
        },
        'train': deepcopy(RF_CONFIG),
        'evaluate': deepcopy(EVALUATE_CONFIG),
    },
    {
        'name': 'slice2_model_xgboost',
        'tags': {
            'study_slice': 'model_type',
            'study_value': 'xgboost',
            'study_hypothesis': 'Ансамблевые модели дают ROC-AUC выше логистической регрессии',
        },
        'process_data': {
            'params': {
                'features': FEATURE_SET_C,
                'train_size': 10000,
            }
        },
        'train': deepcopy(XGB_CONFIG),
        'evaluate': deepcopy(EVALUATE_CONFIG),
    },

    # Разрез 3: набор фич
    {
        'name': 'slice3_features_A',
        'tags': {
            'study_slice': 'features',
            'study_value': 'A',
            'study_hypothesis': 'Добавление информативных числовых и категориальных признаков повышает качество',
        },
        'process_data': {
            'params': {
                'features': FEATURE_SET_A,
                'train_size': 10000,
            }
        },
        'train': deepcopy(RF_CONFIG),
        'evaluate': deepcopy(EVALUATE_CONFIG),
    },
    {
        'name': 'slice3_features_B',
        'tags': {
            'study_slice': 'features',
            'study_value': 'B',
            'study_hypothesis': 'Добавление информативных числовых и категориальных признаков повышает качество',
        },
        'process_data': {
            'params': {
                'features': FEATURE_SET_B,
                'train_size': 10000,
            }
        },
        'train': deepcopy(RF_CONFIG),
        'evaluate': deepcopy(EVALUATE_CONFIG),
    },
    {
        'name': 'slice3_features_C',
        'tags': {
            'study_slice': 'features',
            'study_value': 'C',
            'study_hypothesis': 'Добавление информативных числовых и категориальных признаков повышает качество',
        },
        'process_data': {
            'params': {
                'features': FEATURE_SET_C,
                'train_size': 10000,
            }
        },
        'train': deepcopy(RF_CONFIG),
        'evaluate': deepcopy(EVALUATE_CONFIG),
    },
]


def main():
    original_process_data = read_yaml(PROCESS_DATA_PATH)
    original_train = read_yaml(TRAIN_PATH)
    original_evaluate = read_yaml(EVALUATE_PATH)

    try:
        for idx, experiment in enumerate(EXPERIMENTS, start=1):
            print(f'\n===== [{idx}/{len(EXPERIMENTS)}] {experiment["name"]} =====')

            write_yaml(PROCESS_DATA_PATH, experiment['process_data'])
            write_yaml(TRAIN_PATH, experiment['train'])
            write_yaml(EVALUATE_PATH, experiment['evaluate'])

            env = os.environ.copy()
            env['RUN_NAME'] = experiment['name']
            env['RUN_TAG_SLICE'] = experiment['tags']['study_slice']
            env['RUN_TAG_VALUE'] = experiment['tags']['study_value']
            env['RUN_TAG_HYPOTHESIS'] = experiment['tags']['study_hypothesis']

            subprocess.run(['python', 'runner.py'], check=True, env=env)

    finally:
        write_yaml(PROCESS_DATA_PATH, original_process_data)
        write_yaml(TRAIN_PATH, original_train)
        write_yaml(EVALUATE_PATH, original_evaluate)
        print('\nИсходные конфиги восстановлены.')


if __name__ == '__main__':
    main()
