import pandas as pd
import mlflow
import mlflow.sklearn
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'

MODEL_REGISTRY = {
    'logistic_regression': (LogisticRegression, ['penalty', 'C', 'solver', 'max_iter', 'random_state']),
    'decision_tree': (DecisionTreeClassifier, ['max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']),
    'random_forest': (RandomForestClassifier, ['n_estimators', 'max_depth', 'min_samples_split', 'random_state']),
    'gradient_boosting': (GradientBoostingClassifier, ['n_estimators', 'max_depth', 'learning_rate', 'random_state']),
}


def _get_model_class_and_params(model_type: str):
    model_type = model_type.lower().replace(' ', '_')
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Неизвестный тип модели: {model_type}. "
            f"Доступные: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type]


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Загрузка датасетов')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    logger.info('Датасеты загружены')

    model_type = params.pop('model_type', 'logistic_regression')
    model_class, allowed_params = _get_model_class_and_params(model_type)

    model_params = {
        k: v for k, v in params.items()
        if k in allowed_params
    }
    model_params['random_state'] = RANDOM_STATE

    logger.info(f'Модель: {model_type}, параметры: {model_params}')
    model = model_class(**model_params)
    model.fit(X_train, y_train)

    dump(model, MODEL_FILEPATH)
    logger.info('Модель сохранена')

    if mlflow.active_run():
        mlflow.log_params({
            'model_type': model_type,
            **{f'model_{k}': str(v) for k, v in model_params.items()},
        })
        mlflow.sklearn.log_model(model, 'model', input_example=X_train[:5])


if __name__ == '__main__':
    train()
