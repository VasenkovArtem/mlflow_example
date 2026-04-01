import pandas as pd
from joblib import dump
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params, prepare_params_for_logging

STAGE_NAME = 'train'


def build_model(params: dict):
    params = params.copy()
    model_type = params.pop('model_type')

    if model_type == 'logistic_regression':
        params['random_state'] = RANDOM_STATE
        model = LogisticRegression(**params)

    elif model_type == 'decision_tree':
        params['random_state'] = RANDOM_STATE
        model = DecisionTreeClassifier(**params)

    elif model_type == 'random_forest':
        params['random_state'] = RANDOM_STATE
        model = RandomForestClassifier(**params)

    elif model_type == 'xgboost':
        params['random_state'] = RANDOM_STATE
        params.setdefault('objective', 'binary:logistic')
        params.setdefault('eval_metric', 'logloss')
        model = XGBClassifier(**params)

    else:
        raise ValueError(f'Неизвестный model_type: {model_type}')

    return model_type, model, params


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    X_train = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='X_train'))
    y_train = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='y_train')).squeeze()
    logger.info('Успешно считали датасеты!')

    logger.info('Создаём модель')
    model_type, model, model_params = build_model(params)
    logger.info(f'Тип модели: {model_type}')
    logger.info(f'Параметры модели: {model_params}')

    logger.info('Обучаем модель')
    model.fit(X_train, y_train)

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)
    logger.info('Успешно!')

    if mlflow.active_run():
        mlflow.log_params(prepare_params_for_logging({
            'model_type': model_type,
            **model_params,
        }, prefix='model_'))

        if model_type == 'xgboost':
            mlflow.xgboost.log_model(model, artifact_path='model')
        else:
            mlflow.sklearn.log_model(model, artifact_path='model')
