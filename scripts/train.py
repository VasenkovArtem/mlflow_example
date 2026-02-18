import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

import mlflow
import mlflow.sklearn

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
    model = LogisticRegression(**params)

    logger.info('Обучаем модель')
    model.fit(X_train, y_train)

    logger.info('Сохраняем модель локально')
    dump(model, MODEL_FILEPATH)

    logger.info('Логируем в MLflow')
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_params(params)
    mlflow.sklearn.log_model(model, "model")
    
    logger.info('Шаг Train успешно завершен!')


if __name__ == '__main__':
    train()