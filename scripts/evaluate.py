import os

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import mlflow

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
    y_test_array = y_test.values.ravel()

    logger.info('Начали считать метрики на тесте')
    
    # Calculate all required metrics
    accuracy = accuracy_score(y_test_array, y_pred)
    precision = precision_score(y_test_array, y_pred)
    recall = recall_score(y_test_array, y_pred)
    f1 = f1_score(y_test_array, y_pred)
    roc_auc = roc_auc_score(y_test_array, y_proba)
    pr_auc = average_precision_score(y_test_array, y_proba)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
    }
    
    logger.info(f'Значения метрик - {metrics}')
    
    # Log metrics to MLflow
    mlflow.log_metrics(metrics)
    
    # Create and log artifacts
    logger.info('Создаём артефакты')
    
    # 1. Classification Report
    class_report = classification_report(y_test_array, y_pred)
    with open('/tmp/classification_report.txt', 'w') as f:
        f.write(class_report)
    mlflow.log_artifact('/tmp/classification_report.txt')
    logger.info('  - Classification report сохранён')
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test_array, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('/tmp/confusion_matrix.png')
    mlflow.log_artifact('/tmp/confusion_matrix.png')
    plt.close()
    logger.info('  - Confusion matrix сохранена')
    
    # 3. PR Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test_array, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.4f})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/tmp/pr_curve.png')
    mlflow.log_artifact('/tmp/pr_curve.png')
    plt.close()
    logger.info('  - PR-кривая сохранена')
    
    # 4. Errors CSV
    errors_mask = y_pred != y_test_array
    if errors_mask.sum() > 0:
        errors_df = pd.DataFrame({
            'index': X_test.index[errors_mask],
            'true_label': y_test_array[errors_mask],
            'predicted_label': y_pred[errors_mask],
            'probability': y_proba[errors_mask]
        })
        errors_df.to_csv('/tmp/errors.csv', index=False)
        mlflow.log_artifact('/tmp/errors.csv')
        logger.info(f'  - CSV с ошибками сохранён ({len(errors_df)} ошибок)')
    else:
        logger.info('  - Ошибок не обнаружено, CSV не создан')
    
    logger.info('Все артефакты успешно залогированы!')


if __name__ == '__main__':
    evaluate()