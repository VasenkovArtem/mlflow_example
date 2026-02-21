import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

import mlflow
from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH
from utils import get_logger, load_params

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    get_scorer,
)

from constants import (
    DATASET_PATH_PATTERN,
    MODEL_FILEPATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)

STAGE_NAME = 'evaluate'


def evaluate():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    try:
        current_run = mlflow.active_run()
    except Exception:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

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
        if metric_name in ['roc_auc', 'average_precision']:
            metrics[metric_name] = scorer._score_func(y_test, y_proba)
        else:
            metrics[metric_name] = scorer._score_func(y_test, y_pred)
            
    logger.info(f'Значения метрик - {metrics}')

    mlflow.log_metrics(metrics)
    logger.info('Метрики залогированы в MLflow')

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = metrics.get('average_precision', average_precision_score(y_test, y_proba))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}', linewidth=2)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        pr_curve_path = tmp_file.name
        plt.savefig(pr_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(pr_curve_path, 'pr_curve')
        logger.info('PR-кривая залогирована в MLflow')
        
        os.unlink(pr_curve_path)

if __name__ == '__main__':
    evaluate()