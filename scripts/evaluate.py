import os

import mlflow
import numpy as np
import pandas as pd
from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH
from joblib import load
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from utils import get_logger, load_params


STAGE_NAME = "evaluate"


def evaluate():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info("Начали считывать датасеты")
    splits = [None, None, None, None]
    for i, split_name in enumerate(["X_train", "X_test", "y_train", "y_test"]):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info("Успешно считали датасеты!")

    logger.info("Загружаем обученную модель")
    if not os.path.exists(MODEL_FILEPATH):
        raise FileNotFoundError(
            "Не нашли файл с моделью. Убедитесь, что был запущен шаг с обучением"
        )
    model = load(MODEL_FILEPATH)

    logger.info("Скорим модель на тесте")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = np.where(y_proba >= 0.5, 1, 0)

    logger.info("Начали считать метрики на тесте")

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    metrics['pr_auc'] = auc(recall_curve, precision_curve)

    for metric_name, score in metrics.items():
        mlflow.log_metric(metric_name, float(score))

    report = classification_report(y_test, y_pred)
    with open('/app/data/classification_report.txt', 'w') as f:
        f.write(report)
    mlflow.log_artifact('/app/data/classification_report.txt', artifact_path='artifacts')

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=['true_0', 'true_1'], columns=['pred_0', 'pred_1']).to_csv(
        '/app/data/confusion_matrix.csv', index=True
    )
    mlflow.log_artifact('/app/data/confusion_matrix.csv', artifact_path='artifacts')

if __name__ == "__main__":
    evaluate()
