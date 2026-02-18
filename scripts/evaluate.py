import os
import json

import numpy as np
import pandas as pd
from joblib import load
import mlflow

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'


def _to_1d(y: pd.DataFrame | pd.Series) -> np.ndarray:
    """Приводим y к 1D numpy array."""
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        return y.iloc[:, 0].to_numpy()
    if isinstance(y, pd.Series):
        return y.to_numpy()
    return np.asarray(y).ravel()


def _predict_proba_binary(model, X: pd.DataFrame) -> np.ndarray:
    """
    Возвращает вероятности класса 1.
    Поддерживает:
      - predict_proba
      - decision_function (конвертируем через сигмоиду)
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # proba shape: (n, 2) для бинарной классификации
        return proba[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores).ravel()
        # sigmoid
        return 1.0 / (1.0 + np.exp(-scores))

    raise ValueError(
        "Модель не поддерживает predict_proba и decision_function, "
        "нельзя посчитать ROC-AUC / PR-AUC."
    )


def evaluate():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    # ожидаемые ключи в params/evaluate.yaml:
    # params:
    #   metrics: [accuracy, precision, recall, f1, roc_auc, pr_auc]
    #   threshold: 0.5
    #   reports_dir: reports
    metrics_to_compute = params.get("metrics", ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"])
    threshold = float(params.get("threshold", 0.5))
    reports_dir = params.get("reports_dir", "reports")

    logger.info("Начали считывать датасеты (test)")
    X_test = pd.read_csv(DATASET_PATH_PATTERN.format(split_name="X_test"))
    y_test = pd.read_csv(DATASET_PATH_PATTERN.format(split_name="y_test"))
    y_test_1d = _to_1d(y_test)
    logger.info("Успешно считали датасеты!")

    logger.info("Загружаем обученную модель")
    if not os.path.exists(MODEL_FILEPATH):
        raise FileNotFoundError("Не нашли файл с моделью. Убедитесь, что был запущен шаг с обучением")
    model = load(MODEL_FILEPATH)

    os.makedirs(reports_dir, exist_ok=True)

    logger.info("Скорим модель на тесте")
    y_proba = _predict_proba_binary(model, X_test)
    y_pred = (y_proba >= threshold).astype(int)

    logger.info("Начали считать метрики на тесте")
    metrics = {}

    # считаем только те метрики, которые попросили в конфиге
    for name in metrics_to_compute:
        name_l = name.lower()

        if name_l == "accuracy":
            metrics["accuracy"] = float(accuracy_score(y_test_1d, y_pred))

        elif name_l == "precision":
            metrics["precision"] = float(precision_score(y_test_1d, y_pred, zero_division=0))

        elif name_l == "recall":
            metrics["recall"] = float(recall_score(y_test_1d, y_pred, zero_division=0))

        elif name_l in ("f1", "f1-score", "f1_score"):
            metrics["f1-score"] = float(f1_score(y_test_1d, y_pred, zero_division=0))

        elif name_l in ("roc_auc", "roc-auc", "rocauc"):
            metrics["ROC-AUC"] = float(roc_auc_score(y_test_1d, y_proba))

        elif name_l in ("pr_auc", "prauc", "average_precision", "pr-auc"):
            metrics["PR-AUC"] = float(average_precision_score(y_test_1d, y_proba))

        else:
            raise ValueError(
                f"Неизвестная метрика '{name}'. "
                "Поддерживаются: accuracy, precision, recall, f1, roc_auc, pr_auc"
            )

    logger.info(f"Значения метрик - {metrics}")
    for k, v in metrics.items():
        safe_k = (
            k.replace(" ", "_")
            .replace("-", "_")
            .replace("ROC_AUC", "roc_auc")
            .replace("PR_AUC", "pr_auc")
        )
        mlflow.log_metric(safe_k, float(v))

    # ====== Артефакты ======
    # 1) classification report
    report_text = classification_report(y_test_1d, y_pred, digits=4)
    report_path = os.path.join(reports_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # 2) confusion matrix (png)
    cm = confusion_matrix(y_test_1d, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig = disp.plot().figure_
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    # 3) PR curve (png) — очень полезно для отчёта и задания
    precision, recall, _ = precision_recall_curve(y_test_1d, y_proba)
    fig = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    pr_path = os.path.join(reports_dir, "pr_curve.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close(fig)

    # ====== Сохраним метрики в файл ======
    metrics_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info(f"Сохранили метрики: {metrics_path}")
    logger.info(f"Сохранили артефакты: {report_path}, {cm_path}, {pr_path}")
    mlflow.log_artifacts(reports_dir, artifact_path="reports")


if __name__ == "__main__":
    evaluate()
