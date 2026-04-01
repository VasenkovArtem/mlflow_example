import os

import mlflow
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from constants import ARTIFACTS_DIR, DATASET_PATH_PATTERN, MODEL_FILEPATH
from utils import get_logger, load_params, prepare_params_for_logging

STAGE_NAME = 'evaluate'


def evaluate():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)
    threshold = params.get('threshold', 0.5)

    logger.info('Начали считывать датасеты')
    X_test = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='X_test'))
    y_test = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='y_test')).squeeze()
    logger.info('Успешно считали датасеты!')

    logger.info('Загружаем обученную модель')
    if not os.path.exists(MODEL_FILEPATH):
        raise FileNotFoundError(
            'Не нашли файл с моделью. Убедитесь, что был запущен шаг с обучением'
        )
    model = load(MODEL_FILEPATH)

    logger.info('Считаем предсказания')
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-scores))

    y_pred = (y_proba >= threshold).astype(int)

    logger.info('Начали считать метрики на тесте')
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
    }
    logger.info(f'Значения метрик - {metrics}')

    run_artifacts_dir = ARTIFACTS_DIR
    if mlflow.active_run():
        run_artifacts_dir = os.path.join(ARTIFACTS_DIR, mlflow.active_run().info.run_id)
    os.makedirs(run_artifacts_dir, exist_ok=True)

    # classification report
    report_text = classification_report(y_test, y_pred, digits=4)
    report_path = os.path.join(run_artifacts_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(run_artifacts_dir, 'confusion_matrix.csv')
    pd.DataFrame(
        cm,
        index=['true_0', 'true_1'],
        columns=['pred_0', 'pred_1']
    ).to_csv(cm_path)

    # errors
    errors_mask = y_test != y_pred
    errors_df = X_test.loc[errors_mask].copy()
    errors_df['target'] = y_test.loc[errors_mask].values
    errors_df['prediction'] = y_pred[errors_mask]
    errors_df['proba'] = y_proba[errors_mask]
    errors_path = os.path.join(run_artifacts_dir, 'model_errors.csv')
    errors_df.to_csv(errors_path, index=False)

    # feature importances / coefficients
    fi_path = os.path.join(run_artifacts_dir, 'feature_importances.csv')
    if hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_,
        }).sort_values('importance', ascending=False)
        fi.to_csv(fi_path, index=False)
    elif hasattr(model, 'coef_'):
        coef = np.abs(model.coef_[0])
        fi = pd.DataFrame({
            'feature': X_test.columns,
            'importance': coef,
        }).sort_values('importance', ascending=False)
        fi.to_csv(fi_path, index=False)

    if mlflow.active_run():
        mlflow.log_params(prepare_params_for_logging({
            'threshold': threshold,
            'metrics_requested': params.get('metrics', []),
        }, prefix='eval_'))

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(report_path, artifact_path='evaluation')
        mlflow.log_artifact(cm_path, artifact_path='evaluation')
        mlflow.log_artifact(errors_path, artifact_path='evaluation')

        if os.path.exists(fi_path):
            mlflow.log_artifact(fi_path, artifact_path='evaluation')

    return metrics
