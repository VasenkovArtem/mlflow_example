import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
from joblib import load
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, get_scorer

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
    y_test = np.asarray(y_test).ravel()
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

    if mlflow.active_run():
        mlflow.log_metrics({
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1-score': metrics.get('f1', 0),
            'ROC-AUC': metrics.get('roc_auc', 0),
            'PR-AUC': metrics.get('average_precision', 0),
        })
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        path = os.path.join(tempfile.gettempdir(), 'confusion_matrix.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        mlflow.log_artifact(path, artifact_path='artifacts')
        if os.path.exists(path):
            os.unlink(path)


if __name__ == '__main__':
    evaluate()