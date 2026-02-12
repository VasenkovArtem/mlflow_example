import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import get_scorer

from constants import DATASET_PATH_PATTERN, RUN_ID_FILE
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'

def evaluate():
    logger = get_logger(logger_name=STAGE_NAME)
    eval_params = load_params("evaluate")
    train_params = load_params("train")
    process_params = load_params("process_data")

    logger.info('Начали считывать датасеты')
    splits = []
    for split_name in ['X_train', 'X_test', 'y_train', 'y_test']:
        splits.append(pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name)))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    if not os.path.exists(RUN_ID_FILE):
        raise RuntimeError("run_id.txt not found")
    with open(RUN_ID_FILE, "r") as f:
        run_id = f.read().strip()
    
    logger.info(f"Loading model from MLflow run_id: {run_id}")
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    logger.info('Считаем метрики')
    metrics = {}
    for metric_name in eval_params['params']['metrics']:
        scorer = get_scorer(metric_name)
        score = scorer(model, X_test, y_test)
        metrics[metric_name] = score
    logger.info(f'Метрики: {metrics}')

    mlflow.set_experiment("homework_ulyanova")
    with mlflow.start_run(run_id=run_id):
        
        for name, value in metrics.items():
            mlflow.log_metric(name, value)


        artifact_type = eval_params.get("artifact_type", "classification_report")
        
        if artifact_type == "classification_report":
            report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")
            
        elif artifact_type == "confusion_matrix":
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            plt.close()
            
        elif artifact_type == "feature_importances":
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                fi_dict = {f"feature_{i}": float(v) for i, v in enumerate(fi)}
                mlflow.log_dict(fi_dict, "feature_importances.json")
                
        elif artifact_type == "pr_curve":
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            plt.figure()
            plt.plot(recall, precision, label=f'AP = {ap:.2f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('PR Curve')
            plt.legend()
            plt.savefig("pr_curve.png")
            mlflow.log_artifact("pr_curve.png")
            plt.close()
            
        elif artifact_type == "errors_csv":
            X_test_df = pd.read_csv(DATASET_PATH_PATTERN.format(split_name="X_test"))
            errors_mask = (y_pred != y_test.values.flatten())
            errors_df = X_test_df[errors_mask].copy()
            errors_df["y_true"] = y_test.values[errors_mask].flatten()
            errors_df["y_pred"] = y_pred[errors_mask]
            errors_df["y_proba"] = y_proba[errors_mask]
            errors_df.to_csv("model_errors.csv", index=False)
            mlflow.log_artifact("model_errors.csv")

    logger.info(f'Завершена оценка для run_id: {run_id}')