import os

import numpy as np
import pandas as pd
from joblib import load
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
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import mlflow

from constants import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    DATASET_PATH_PATTERN,
    ERRORS_CSV_PATH,
    MODEL_FILEPATH,
    PR_CURVE_PATH,
)
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'


def evaluate():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    _, X_test, _, y_test = splits
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

    metrics = {
        'accuracy': accuracy_score(y_test_array, y_pred),
        'precision': precision_score(y_test_array, y_pred),
        'recall': recall_score(y_test_array, y_pred),
        'f1': f1_score(y_test_array, y_pred),
        'roc_auc': roc_auc_score(y_test_array, y_proba),
        'pr_auc': average_precision_score(y_test_array, y_proba),
    }

    logger.info(f'Значения метрик - {metrics}')

    to_log = params.get('metrics')
    if to_log:
        mlflow.log_metrics({name: metrics[name] for name in to_log if name in metrics})
    else:
        mlflow.log_metrics(metrics)

    logger.info('Создаём артефакты')

    class_report = classification_report(y_test_array, y_pred)
    with open(CLASSIFICATION_REPORT_PATH, 'w') as f:
        f.write(class_report)
    mlflow.log_artifact(CLASSIFICATION_REPORT_PATH)
    logger.info('  - Classification report сохранён')

    cm = confusion_matrix(y_test_array, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    mlflow.log_artifact(CONFUSION_MATRIX_PATH)
    plt.close()
    logger.info('  - Confusion matrix сохранена')

    pr_auc = metrics['pr_auc']
    precision_vals, recall_vals, _ = precision_recall_curve(y_test_array, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.4f})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PR_CURVE_PATH)
    mlflow.log_artifact(PR_CURVE_PATH)
    plt.close()
    logger.info('  - PR-кривая сохранена')

    errors_mask = y_pred != y_test_array
    if errors_mask.sum() > 0:
        errors_df = X_test.loc[errors_mask].copy()
        errors_df['true_label'] = y_test_array[errors_mask]
        errors_df['predicted_label'] = y_pred[errors_mask]
        errors_df['probability'] = y_proba[errors_mask]
        errors_df.to_csv(ERRORS_CSV_PATH, index=False)
        mlflow.log_artifact(ERRORS_CSV_PATH)
        logger.info(f'  - CSV с ошибками сохранён ({len(errors_df)} ошибок)')
    else:
        logger.info('  - Ошибок не обнаружено, CSV не создан')

    logger.info('Все артефакты успешно залогированы!')


if __name__ == '__main__':
    evaluate()
