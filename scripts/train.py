import mlflow
import time

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    logger.info('Создаём модель')
    params['random_state'] = RANDOM_STATE

    mlflow.log_param("model_type", params["model_type"])
    mlflow.log_param("random_state", params['random_state'])
    mlflow.log_params(params[params["model_type"]])

    logger.info(f'    Параметры модели: {params}')
    if params["model_type"] == "log_reg":
        logger.info('Используем логистическую регрессию')
        model = LogisticRegression(**params["log_reg"])
    elif params["model_type"] == "random_forest":
        logger.info('Используем случайный лес')
        model = RandomForestClassifier(**params["random_forest"])
    elif params["model_type"] == "grad_boosting":
        logger.info('Используем градиентный бустинг')
        model = GradientBoostingClassifier(**params["grad_boosting"])
    else:
        raise ValueError("Странный вид модели")
    

    logger.info('Обучаем модель')
    model.fit(X_train, y_train)

    logger.info('Сохраняем модель')
    
    timestamp = int(time.time())
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=f"homework_model_{timestamp}",
        input_example=X_train[:5]
    )
    dump(model, MODEL_FILEPATH)
    logger.info('Успешно!')


if __name__ == '__main__':
    train()
