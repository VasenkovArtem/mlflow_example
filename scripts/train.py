import pandas as pd
import mlflow
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)
    mlflow.log_params(params)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    logger.info('Создаём модель')
    params['random_state'] = RANDOM_STATE
    logger.info(f'    Параметры модели: {params}')

    if params['model'] == 'LogisticRegression':
        model = LogisticRegression(**params['hyperparams'])
    elif params['model'] == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(**params['hyperparams'])

    logger.info('Обучаем модель')
    model.fit(X_train, y_train)

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)
    mlflow.sklearn.log_model(model, 'model')
    logger.info('Успешно!')


if __name__ == '__main__':
    train()
