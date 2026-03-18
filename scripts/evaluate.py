import os

import mlflow
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    get_scorer,
)

from constants import ARTIFACT_DIR, DATASET_PATH_PATTERN, MODEL_FILEPATH
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'

SCORER_METRICS = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')


def _ensure_artifact_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


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

    y_test_flat = np.asarray(y_test).ravel()
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    y_pred = np.asarray(y_pred).ravel()

    logger.info('Начали считать метрики на тесте')
    metrics = {}
    for metric_name in params.get('metrics', SCORER_METRICS):
        if metric_name not in SCORER_METRICS:
            continue
        scorer = get_scorer(metric_name)
        metrics[metric_name] = float(scorer(model, X_test, y_test))
    metrics['pr_auc'] = float(
        average_precision_score(y_test_flat, y_proba)
    )
    logger.info(f'Значения метрик - {metrics}')

    artifact_type = params.get('artifact_type', 'classification_report')
    _ensure_artifact_dir()
    artifact_path = None

    if artifact_type == 'classification_report':
        path = os.path.join(ARTIFACT_DIR, 'classification_report.txt')
        text = classification_report(y_test_flat, y_pred)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        artifact_path = path

    elif artifact_type == 'confusion_matrix':
        path = os.path.join(ARTIFACT_DIR, 'confusion_matrix.png')
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_test_flat, y_pred)
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        artifact_path = path

    elif artifact_type == 'feature_importances':
        path = os.path.join(ARTIFACT_DIR, 'feature_importances.csv')
        n_features = X_test.shape[1]
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        elif hasattr(model, 'coef_'):
            imp = np.abs(np.asarray(model.coef_).ravel())
            if len(imp) != n_features:
                imp = np.pad(imp, (0, max(0, n_features - len(imp))))[:n_features]
        else:
            imp = np.ones(n_features) / n_features
        pd.DataFrame({'feature_index': range(len(imp)), 'importance': imp}).to_csv(
            path, index=False
        )
        artifact_path = path

    elif artifact_type == 'errors_csv':
        path = os.path.join(ARTIFACT_DIR, 'prediction_errors.csv')
        err_mask = y_pred != y_test_flat
        df_err = pd.DataFrame(
            {
                'y_true': y_test_flat[err_mask],
                'y_pred': y_pred[err_mask],
                'proba_positive': y_proba[err_mask],
            }
        )
        df_err.to_csv(path, index=False)
        artifact_path = path

    elif artifact_type == 'pr_curve':
        path = os.path.join(ARTIFACT_DIR, 'pr_curve.png')
        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(
            y_test_flat, y_proba, ax=ax
        )
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        artifact_path = path
    else:
        logger.warning(f'Неизвестный artifact_type={artifact_type}, используем classification_report')
        path = os.path.join(ARTIFACT_DIR, 'classification_report.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(classification_report(y_test_flat, y_pred))
        artifact_path = path

    if mlflow.active_run():
        mlflow.log_metrics(metrics)
        if artifact_path and os.path.isfile(artifact_path):
            mlflow.log_artifact(artifact_path)


if __name__ == '__main__':
    evaluate()
