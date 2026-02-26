import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params
import mlflow

STAGE_NAME = 'train'

MODEL_CLASSES = {
    'LogisticRegression': LogisticRegression,
    'DecisionTree': DecisionTreeClassifier,
    'RandomForest': RandomForestClassifier,
    'GradientBoosting': GradientBoostingClassifier,
}


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    model_type = params.pop('model_type', 'LogisticRegression')
    model_class = MODEL_CLASSES[model_type]

    params['random_state'] = RANDOM_STATE
    logger.info(f'Создаём модель: {model_type}')
    logger.info(f'    Параметры модели: {params}')
    model = model_class(**params)

    logger.info('Обучаем модель')
    model.fit(X_train, y_train.values.ravel())

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)
    logger.info('Успешно!')

    train_dataset = pd.concat([X_train, y_train], axis=1)
    train_dataset.to_csv("train_dataset.csv", index=False)
    mlflow.log_artifact("train_dataset.csv", artifact_path="datasets")
    logger.info('Датасет залогирован в MLflow')
    logger.info('Успешно!')


if __name__ == '__main__':
    train()
