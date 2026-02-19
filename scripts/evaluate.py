import os

import mlflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import get_scorer, classification_report

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'


def evaluate(params_filepath: str | None = None):
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME, params_filepath=params_filepath)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')
    
    logger.info('Загружаем обученную модель')
    if not os.path.exists(MODEL_FILEPATH):
        raise FileNotFoundError(
            'Не нашли файл с моделью. Убедитесь, что был запущен шаг с обучением'
        )
    model = load(MODEL_FILEPATH)

    logger.info('Скорим модель на тесте')
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = np.where(y_proba >= 0.5, 1, 0)

    y_test_array = np.asarray(y_test).reshape(-1)

    logger.info('Начали считать метрики на тесте')
    metrics = {}
    for metric_name in params['metrics']:
        scorer = get_scorer(metric_name)
        score = scorer(model, X_test, y_test)
        metrics[metric_name] = score
    logger.info(f'Значения метрик - {metrics}')

    mlflow.log_metrics(metrics)

    # add artifact
    class_report = classification_report(y_test_array, y_pred)
    with open('/tmp/classification_report.txt', 'w') as f:
        f.write(class_report)
    mlflow.log_artifact('/tmp/classification_report.txt')
    logger.info('Classification report saved')

if __name__ == '__main__':
    evaluate()