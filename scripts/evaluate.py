import os
import tempfile
from contextlib import nullcontext
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
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from constants import (
    DATASET_PATH_PATTERN,
    MODEL_FILEPATH,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'


def _log_artifacts(y_test, y_pred, y_proba):
    report = classification_report(y_test, y_pred)
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt') as tmp:
        tmp.write(report)
        report_path = tmp.name
    mlflow.log_artifact(report_path, artifact_path='artifacts')

    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
    cm_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
    fig_cm.savefig(cm_path, bbox_inches='tight')
    plt.close(fig_cm)
    mlflow.log_artifact(cm_path, artifact_path='artifacts')

    fig_pr, ax_pr = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax_pr)
    pr_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
    fig_pr.savefig(pr_path, bbox_inches='tight')
    plt.close(fig_pr)
    mlflow.log_artifact(pr_path, artifact_path='artifacts')


def evaluate(start_new_run: bool = True, run_name: str = 'evaluate_model'):
    logger = get_logger(logger_name=STAGE_NAME)
    eval_params = load_params(stage_name=STAGE_NAME)
    data_params = load_params(stage_name='process_data')

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    y_test = y_test.squeeze()
    logger.info('Успешно считали датасеты!')

    logger.info('Загружаем обученную модель')
    if not os.path.exists(MODEL_FILEPATH):
        raise FileNotFoundError('Не найден файл с моделью. Запустите train.py')
    model = load(MODEL_FILEPATH)

    if not start_new_run and mlflow.active_run() is None:
        raise RuntimeError('Нет активного MLflow ранa, но start_new_run=False')

    run_ctx = mlflow.start_run(run_name=run_name) if start_new_run else nullcontext()

    with run_ctx:
        mlflow.log_param('stage', STAGE_NAME)
        mlflow.log_param('data_features', data_params['features'])
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))
        mlflow.log_param('requested_metrics', eval_params['metrics'])

        logger.info('Начали считать метрики на тесте')
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba),
        }

        selected_metrics = {k: v for k, v in metrics.items() if k in eval_params['metrics']}

        mlflow.log_metrics(selected_metrics)
        logger.info(f'Значения метрик - {selected_metrics}')

        _log_artifacts(y_test, y_pred, y_proba)


if __name__ == '__main__':
    evaluate()