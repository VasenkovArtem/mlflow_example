import os

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import (
    get_scorer,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH
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

    logger.info('Скорим модель на тесте')
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = np.where(y_proba >= 0.5, 1, 0)

    logger.info('Начали считать метрики на тесте')
    metrics = {}
    for metric_name in params['metrics']:
        scorer = get_scorer(metric_name)
        score = scorer(model, X_test, y_test)
        metrics[metric_name] = score
    logger.info(f'Значения метрик - {metrics}')

    mlflow.log_metrics(metrics)

    # Артефакт 1: Classification Report
    logger.info('Сохраняем classification report')
    report = classification_report(y_test, y_pred)
    report_path = '/app/data/classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # Артефакт 2: Confusion Matrix
    logger.info('Сохраняем confusion matrix')
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    cm_path = '/app/data/confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Артефакт 3: CSV с ошибками модели
    logger.info('Сохраняем CSV с ошибками модели')
    errors_mask = y_pred != y_test.values.ravel()
    errors_df = pd.DataFrame({
        'y_true': y_test.values.ravel()[errors_mask],
        'y_pred': y_pred[errors_mask],
        'y_proba': y_proba[errors_mask],
    })
    errors_path = '/app/data/errors.csv'
    errors_df.to_csv(errors_path, index=False)
    mlflow.log_artifact(errors_path)


if __name__ == '__main__':
    evaluate()
