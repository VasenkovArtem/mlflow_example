import os

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import get_scorer
from sklearn.metrics import classification_report

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'


def evaluate(mlflow, run):
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')
    
    # Логируем в MLflow
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_artifact(DATASET_PATH_PATTERN.format(split_name='X_train'), artifact_path='datasets')
        mlflow.log_artifact(DATASET_PATH_PATTERN.format(split_name='X_test'), artifact_path='datasets')
        mlflow.log_artifact(DATASET_PATH_PATTERN.format(split_name='y_train'), artifact_path='datasets')
        mlflow.log_artifact(DATASET_PATH_PATTERN.format(split_name='y_test'), artifact_path='datasets')
    
    logger.info('Загружаем обученную модель')
    if not os.path.exists(MODEL_FILEPATH):
        raise FileNotFoundError(
            'Не нашли файл с моделью. Убедитесь, что был запущен шаг с обучением'
        )
    model = load(MODEL_FILEPATH)

    logger.info('Скорим модель на тесте')
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = np.where(y_proba >= 0.5, 1, 0)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    with open('classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    
    # Логируем в MLflow
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_artifact('classification_report.txt')
    
    logger.info('Начали считать метрики на тесте')
    metrics = {}
    for metric_name in params['metrics']:
        scorer = get_scorer(metric_name)
        score = scorer(model, X_test, y_test)
        metrics[metric_name] = score
    logger.info(f'Значения метрик - {metrics}')
    
    mlflow.log_metrics({'Accuracy': metrics['accuracy'],'Precision': metrics['precision'],'Recall': metrics['recall'], 'F1': metrics['f1'], 'ROC-AUC': metrics['roc_auc'], 'PR-AUC': metrics['average_precision']}, run_id=run.info.run_id)


if __name__ == '__main__':
    evaluate()