import os
import tempfile

import mlflow
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
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

    logger.info('Считаем вероятности и предсказания на тесте')
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    logger.info('Начали считать метрики на тесте')
    metrics = {}

    if 'accuracy' in params['metrics']:
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
    if 'precision' in params['metrics']:
        metrics['precision'] = precision_score(y_test, y_pred)
    if 'recall' in params['metrics']:
        metrics['recall'] = recall_score(y_test, y_pred)
    if 'f1' in params['metrics']:
        metrics['f1'] = f1_score(y_test, y_pred)
    if 'roc_auc' in params['metrics']:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    if 'pr_auc' in params['metrics']:
        metrics['pr_auc'] = average_precision_score(y_test, y_proba)

    logger.info(f'Значения метрик - {metrics}')

    # логируем метрики в MLflow
    if mlflow.active_run() is not None:
        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        # создаём и логируем артефакт: classification report и PR-кривая
        with tempfile.TemporaryDirectory() as tmpdir:
            # classification report
            report_path = os.path.join(tmpdir, 'classification_report.txt')
            with open(report_path, 'w') as f:
                f.write(classification_report(y_test, y_pred))

            # PR-кривая
            import matplotlib.pyplot as plt

            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            plt.figure()
            plt.step(recall, precision, where='post')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve')
            pr_curve_path = os.path.join(tmpdir, 'pr_curve.png')
            plt.savefig(pr_curve_path, bbox_inches='tight')
            plt.close()

            mlflow.log_artifacts(tmpdir, artifact_path='evaluation_artifacts')


if __name__ == '__main__':
    evaluate()