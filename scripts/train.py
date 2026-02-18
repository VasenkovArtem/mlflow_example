import pandas as pd
from joblib import dump
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'


def _build_model(model_type: str, model_params: dict):
    model_type = model_type.lower()

    if model_type == 'logreg':
        return LogisticRegression(**model_params)

    if model_type == 'decision_tree':
        return DecisionTreeClassifier(**model_params)

    if model_type == 'random_forest':
        return RandomForestClassifier(**model_params)

    if model_type == 'gradient_boosting':
        return GradientBoostingClassifier(**model_params)

    raise ValueError(
        f'Неизвестный model_type="{model_type}". '
        f'Ожидается: logreg, decision_tree, random_forest, gradient_boosting'
    )


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    # ожидаем структуру конфига:
    # params:
    #   model_type: logreg
    #   model_params: {...}
    model_type = params.get('model_type', 'logreg')
    model_params = params.get('model_params', {})
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("model_params",  model_params)

    logger.info('Начали считывать датасеты')
    X_train = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='X_train'))
    y_train = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='y_train'))
    logger.info('Успешно считали датасеты!')

    # приводим y_train к 1D (Series)
    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]

    # прокидываем random_state, если параметр есть у модели
    if 'random_state' not in model_params:
        model_params['random_state'] = RANDOM_STATE

    logger.info('Создаём модель')
    logger.info(f'    model_type: {model_type}')
    logger.info(f'    model_params: {model_params}')

    model = _build_model(model_type=model_type, model_params=model_params)

    logger.info('Обучаем модель')
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
    )

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)
    logger.info('Успешно!')


if __name__ == '__main__':
    train()
