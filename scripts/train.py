import os
import tempfile
from contextlib import nullcontext
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from constants import (
    DATASET_PATH_PATTERN,
    MODEL_FILEPATH,
    RANDOM_STATE,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from utils import get_logger, load_params
STAGE_NAME = 'train'


def _build_model(train_params: dict):
    model_type = train_params['model_type']

    if model_type == 'logistic_regression':
        cfg = train_params['logistic_regression']
        model = LogisticRegression(
            penalty=cfg['penalty'],
            C=cfg['C'],
            solver=cfg['solver'],
            max_iter=cfg['max_iter'],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        used_params = cfg
    elif model_type == 'random_forest':
        cfg = train_params['random_forest']
        model = RandomForestClassifier(
            n_estimators=cfg['n_estimators'],
            max_depth=cfg['max_depth'],
            min_samples_split=cfg['min_samples_split'],
            min_samples_leaf=cfg['min_samples_leaf'],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        used_params = cfg
    elif model_type == 'gradient_boosting':
        cfg = train_params['gradient_boosting']
        model = GradientBoostingClassifier(
            n_estimators=cfg['n_estimators'],
            learning_rate=cfg['learning_rate'],
            max_depth=cfg['max_depth'],
            random_state=RANDOM_STATE,
        )
        used_params = cfg
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model_type, used_params, model


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

    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax_roc)
    roc_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
    fig_roc.savefig(roc_path, bbox_inches='tight')
    plt.close(fig_roc)
    mlflow.log_artifact(roc_path, artifact_path='artifacts')


def train(start_new_run: bool = True, run_name: str | None = None):
    logger = get_logger(logger_name=STAGE_NAME)
    train_params = load_params(stage_name=STAGE_NAME)
    data_params = load_params(stage_name='process_data')

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    logger.info('Успешно считали датасеты!')

    logger.info('Начали обучение модели')
    model_type, model_cfg, model = _build_model(train_params)
    model.fit(X_train, y_train)
    logger.info('Модель успешно обучена')

    os.makedirs(os.path.dirname(MODEL_FILEPATH) or '.', exist_ok=True)
    dump(model, MODEL_FILEPATH)
    logger.info(f'Сохранили модель в {MODEL_FILEPATH}')

    if not start_new_run and mlflow.active_run() is None:
        raise RuntimeError('Нет активного MLflow ранa, но start_new_run=False')

    resolved_run_name = run_name or f'train_{model_type}'
    run_ctx = mlflow.start_run(run_name=resolved_run_name) if start_new_run else nullcontext()

    with run_ctx:
        mlflow.log_param('data_features', data_params['features'])
        mlflow.log_param('train_size', len(y_train))
        mlflow.log_param('test_size', len(y_test))
        mlflow.log_param('model_type', model_type)
        mlflow.log_params(model_cfg)
        mlflow.log_param('random_state', RANDOM_STATE)

        for split_name in ['X_train', 'X_test', 'y_train', 'y_test']:
            split_path = DATASET_PATH_PATTERN.format(split_name=split_name)
            if os.path.exists(split_path):
                mlflow.log_artifact(split_path, artifact_path='data')

        logger.info('Считаем метрики на тесте')
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
        mlflow.log_metrics(metrics)
        logger.info(f'Значения метрик: {metrics}')

        _log_artifacts(y_test, y_pred, y_proba)

        mlflow.sklearn.log_model(model, artifact_path='model')
        logger.info('Модель и артефакты залогированы в MLflow')


if __name__ == '__main__':
    train()
