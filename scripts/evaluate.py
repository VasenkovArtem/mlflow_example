import os

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import get_scorer

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH
from utils import get_logger, load_params

from setup import log_mlflow
import mlflow

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve
)


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
    scores = {
        f"metric_{metric_name}": get_scorer(metric_name)(model, X_test, y_test) 
        for metric_name in params['metrics']
    }
    log_mlflow(**scores)
    logger.info(f'Значения метрик - {scores}')

    logger.info('Создаём classification report')
    report = classification_report(y_test, y_pred, zero_division=0)
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)
    
    logger.info('Создаём confusion matrix')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, dpi=100)
    plt.close()
    mlflow.log_artifact(cm_path)
    
    logger.info('Создаём PR-кривую')
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, linewidth=2, color='blue')
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve (AUC = {scores['metric_average_precision']:.3f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = "pr_curve.png"
    plt.savefig(pr_path, dpi=100)
    plt.close()
    mlflow.log_artifact(pr_path)
    
    logger.info('Создаём CSV с ошибками модели')
    errors_mask = y_pred != y_test.values.ravel()
    errors_df = X_test[errors_mask].copy()
    errors_df['y_true'] = y_test[errors_mask].values
    errors_df['y_pred'] = y_pred[errors_mask]
    errors_df['y_proba'] = y_proba[errors_mask]
    errors_path = "model_errors.csv"
    errors_df.to_csv(errors_path, index=False)
    mlflow.log_artifact(errors_path)
    
    logger.info(f'Всего ошибок: {errors_mask.sum()} из {len(y_test)}')
    logger.info('Все метрики и артефакты залогированы в MLflow!')


if __name__ == '__main__':
    evaluate()