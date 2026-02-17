import os

import numpy as np
import pandas as pd
import mlflow
from joblib import load
from sklearn.metrics import get_scorer, ConfusionMatrixDisplay

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, ARTIFACTS_PATH
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'


def evaluate():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

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


    logger.info('Начали считать метрики на тесте')
    metrics = {}
    for metric_name in params['metrics']:
        scorer = get_scorer(metric_name)
        score = scorer(model, X_test, y_test)
        metrics[metric_name] = score

    mlflow.log_metrics(metrics)
    
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )
    filename = ARTIFACTS_PATH + 'cm.png'
    disp.figure_.savefig(filename)
    mlflow.log_artifact(filename)

    logger.info(f'Значения метрик - {metrics}')


if __name__ == '__main__':
    evaluate()
