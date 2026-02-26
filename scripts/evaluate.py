import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    get_scorer,
    precision_recall_curve,
)

from constants import ARTIFACTS_DIR, DATASET_PATH_PATTERN, MODEL_FILEPATH
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
    y_test_values = y_test.values.ravel()

    logger.info("Начали считать метрики на тесте")
    metrics = {}
    for metric_name in params["metrics"]:
        scorer = get_scorer(metric_name)
        score = scorer(model, X_test, y_test_values)
        metrics[metric_name] = score
    logger.info(f"Значения метрик - {metrics}")

    logger.info("Генерируем артефакты")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Classification report
    report = classification_report(y_test_values, y_pred)
    report_path = os.path.join(ARTIFACTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test_values, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    # PR curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test_values, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_vals, precision_vals)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    pr_path = os.path.join(ARTIFACTS_DIR, "pr_curve.png")
    fig.savefig(pr_path, bbox_inches="tight")
    plt.close(fig)

    # Errors CSV
    mask = y_pred != y_test_values
    errors_df = pd.DataFrame(X_test.values[mask], columns=X_test.columns)
    errors_df["y_true"] = y_test_values[mask]
    errors_df["y_pred"] = y_pred[mask]
    errors_df["y_proba"] = y_proba[mask]
    errors_path = os.path.join(ARTIFACTS_DIR, "errors.csv")
    errors_df.to_csv(errors_path, index=False)

    logger.info("Артефакты сохранены!")

    return metrics


if __name__ == "__main__":
    evaluate()
