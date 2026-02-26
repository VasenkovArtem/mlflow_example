import os
import tempfile

import numpy as np
import pandas as pd
import mlflow
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'


def evaluate():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Загрузка датасетов')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    X_test = X_test.values
    y_test = y_test.values.ravel()
    logger.info('Датасеты загружены')

    if not os.path.exists(MODEL_FILEPATH):
        raise FileNotFoundError(
            'Файл модели не найден. Сначала выполните шаг обучения.'
        )
    model = load(MODEL_FILEPATH)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = np.where(y_proba >= 0.5, 1, 0)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0,
        'pr_auc': average_precision_score(y_test, y_proba),
    }
    logger.info(f'Метрики: {metrics}')

    if mlflow.active_run():
        mlflow.log_metrics(metrics)

        artifact_type = params.get('artifact_type', 'classification_report')

        if artifact_type == 'classification_report':
            report = classification_report(y_test, y_pred, target_names=['<=50K', '>50K'])
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(report)
                mlflow.log_artifact(f.name, 'classification_report.txt')
                os.unlink(f.name)

        elif artifact_type == 'confusion_matrix':
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(cm, cmap='Blues')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['<=50K', '>50K'])
            ax.set_yticklabels(['<=50K', '>50K'])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(f.name, 'confusion_matrix.png')
                os.unlink(f.name)

        elif artifact_type == 'feature_importances':
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                n_features = min(len(importances), 20)
                indices = np.argsort(importances)[::-1][:n_features]
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(range(n_features), importances[indices])
                ax.set_yticks(range(n_features))
                ax.set_yticklabels([f'Feature {i}' for i in indices])
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importances')
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    plt.savefig(f.name, bbox_inches='tight')
                    plt.close()
                    mlflow.log_artifact(f.name, 'feature_importances.png')
                    os.unlink(f.name)
            else:
                logger.warning('У модели нет feature_importances_, логируем classification_report')
                report = classification_report(y_test, y_pred, target_names=['<=50K', '>50K'])
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(report)
                    mlflow.log_artifact(f.name, 'classification_report.txt')
                    os.unlink(f.name)

        elif artifact_type == 'errors_csv':
            errors_df = pd.DataFrame({
                'y_true': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'is_error': y_test != y_pred,
            })
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                errors_df.to_csv(f.name, index=False)
                mlflow.log_artifact(f.name, 'model_errors.csv')
                os.unlink(f.name)

        elif artifact_type == 'pr_curve':
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(recall, precision)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(f.name, 'pr_curve.png')
                os.unlink(f.name)


if __name__ == '__main__':
    evaluate()
