import pandas as pd
from joblib import dump
import mlflow
import os
import tempfile
from models import instantiate_model

from constants import (
    DATASET_PATH_PATTERN,
    MODEL_FILEPATH,
    RANDOM_STATE,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)
from utils import get_logger, load_params

STAGE_NAME = 'train'


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    try:
        current_run = mlflow.active_run()
    except Exception:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')



    logger.info('Создаём модель')

    params['model_params']['random_state'] = RANDOM_STATE
    logger.info(f'    Параметры модели: {params["model_params"]}')
    model = instantiate_model(params['model'], params['model_params'])

    mlflow.log_param('model_type', params['model'])
    
    for key, value in params['model_params'].items():
        mlflow.log_param(f'model_{key}', value)

    logger.info('Логируем датасет X_train в MLflow')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        X_train_path = tmp_file.name
        X_train_df = pd.DataFrame(X_train)
        X_train_df.to_csv(X_train_path, index=False)
        
        mlflow.log_artifact(X_train_path, 'datasets')
        logger.info(f'Датасет X_train залогирован: {X_train.shape[0]} строк, {X_train.shape[1]} признаков')
        
        os.unlink(X_train_path)
    
    mlflow.log_param('train_random_state', RANDOM_STATE)
    mlflow.log_param('train_dataset_size', len(X_train))

    logger.info('Обучаем модель')
    model.fit(X_train, y_train)

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)

    mlflow.sklearn.log_model(model, "model")
    
    logger.info('Успешно!')


if __name__ == '__main__':
    train()
