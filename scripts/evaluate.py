import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import mlflow
from joblib import load
from sklearn.metrics import get_scorer, confusion_matrix

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'


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

    logger.info('Начали считать метрики на тесте')
    metrics = {}
    for metric_name in params['metrics']:
        scorer = get_scorer(metric_name)
        score = scorer(model, X_test, y_test.squeeze())
        metrics[metric_name] = score
    logger.info(f'Значения метрик - {metrics}')
    mlflow.log_metrics(metrics)

    y_pred = model.predict(X_test)
    y_true = y_test.squeeze()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('confusion matrix')
    plt.ylabel('истинный класс')
    plt.xlabel('предсказанный класс')
    cm_path = 'cm.png'
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    errors_df = X_test.copy()
    errors_df['true_label'] = y_true.values
    errors_df['predicted_label'] = y_pred
    errors_df = errors_df[errors_df['true_label'] != errors_df['predicted_label']]
    errors_path = 'errors.csv'
    errors_df.to_csv(errors_path, index=False)
    mlflow.log_artifact(errors_path)


if __name__ == '__main__':
    evaluate()