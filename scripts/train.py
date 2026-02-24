import pandas as pd
import mlflow
import mlflow.sklearn
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'

def make_model(model_type: str, params: dict):
    if model_type == "logreg":
        model_params = dict(params.get("logreg", {}))
        model_params["random_state"] = RANDOM_STATE
        model = LogisticRegression(**model_params)
        return model, model_params, "LogisticRegression"

    if model_type == "dtree":
        model_params = dict(params.get("dtree", {}))
        model_params["random_state"] = RANDOM_STATE
        model = DecisionTreeClassifier(**model_params)
        return model, model_params, "DecisionTreeClassifier"

    if model_type == "rf":
        model_params = dict(params.get("rf", {}))
        model_params["random_state"] = RANDOM_STATE
        model = RandomForestClassifier(**model_params)
        return model, model_params, "RandomForestClassifier"

def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    logger.info('Создаём модель')

    model_type = params.get("model_type", "logreg")
    model, model_params, model_class_name = make_model(model_type, params)
    mlflow.log_param("model_class", model_class_name)
    mlflow.log_params({f"model__{k}": v for k, v in model_params.items()})

    logger.info(f'    Параметры модели: {model_params}')

    logger.info('Обучаем модель')
    model.fit(X_train, y_train)

    logger.info('Сохраняем модель')
    mlflow.sklearn.log_model(model, artifact_path="model")
    dump(model, MODEL_FILEPATH)
    logger.info('Успешно!')


if __name__ == '__main__':
    train()
