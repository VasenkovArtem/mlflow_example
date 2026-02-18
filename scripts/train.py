import pandas as pd
import mlflow
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'

MODEL_REGISTRY = {
    'logistic_regression': (
        LogisticRegression,
        {'penalty', 'C', 'solver', 'max_iter'},
    ),
    'decision_tree': (
        DecisionTreeClassifier,
        {'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf'},
    ),
    'random_forest': (
        RandomForestClassifier,
        {'n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf'},
    ),
    'xgboost': (
        XGBClassifier,
        {'n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree'},
    ),
}


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    model_type = params.get('model_type', 'logistic_regression')
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f'Неизвестный model_type: {model_type}. Доступны: {list(MODEL_REGISTRY)}')
    model_class, param_names = MODEL_REGISTRY[model_type]
    model_params = {k: v for k, v in params.items() if k != 'model_type' and k in param_names}
    model_params['random_state'] = RANDOM_STATE

    logger.info('Создаём модель')
    logger.info(f'    type: {model_type}, params: {model_params}')
    model = model_class(**model_params)

    logger.info('Обучаем модель')
    model.fit(X_train, y_train)

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)

    if mlflow.active_run():
        mlflow.log_params({'model_type': model_type, **model_params})
        mlflow.sklearn.log_model(model, artifact_path='model')

    logger.info('Успешно!')


if __name__ == '__main__':
    train()
