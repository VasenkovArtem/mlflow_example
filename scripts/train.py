import pandas as pd
from joblib import dump

import mlflow
import mlflow.sklearn


from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


STAGE_NAME = 'train'


def train(params_filepath: str | None = None):
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME, params_filepath=params_filepath)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    models = {
        'LogisticRegression': LogisticRegression,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'XGBClassifier': XGBClassifier,
    }

    if params['model_type'] not in models:
        raise ValueError(f'Model {params["model_type"]} not found. Valid models are: {list(models.keys())}')

    model_params = {k: v for k,v in params.items() if k != 'model_type'}
    model = models[params['model_type']](**model_params)


    logger.info('Обучаем модель')
    mlflow.log_param('model_type', params['model_type'])
    mlflow.log_params(model_params)
    model.fit(X_train, y_train)

   

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)
    logger.info('Успешно!')

     # Log model to MLflow
    mlflow.sklearn.log_model(model, 'model')


if __name__ == '__main__':
    train()
