import mlflow
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = "train"


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info("Начали считывать датасеты")
    splits = [None, None, None, None]
    for i, split_name in enumerate(["X_train", "X_test", "y_train", "y_test"]):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info("Успешно считали датасеты!")

    mlflow.log_artifact(
        DATASET_PATH_PATTERN.format(split_name='X_train'),
        artifact_path='datasets'
    )
    mlflow.log_artifact(
        DATASET_PATH_PATTERN.format(split_name='y_train'),
        artifact_path='datasets'
    )

    logger.info("Создаём модель")
    model_type = params.pop('model_type')
    params["random_state"] = RANDOM_STATE
    logger.info(f"    Параметры модели: {params}")

    if model_type == 'logreg':
        model = LogisticRegression(**params)
    elif model_type == 'dt':
        model = DecisionTreeClassifier(**params)
    elif model_type == 'rf':
        model = RandomForestClassifier(**params)
    elif model_type == 'xgb':
        model = XGBClassifier(**params)
    else:
        raise ValueError(f'Неизвестный тип модели: {model_type}')

    logger.info("Обучаем модель")
    model.fit(X_train, y_train)

    mlflow.log_param('model.type', model_type)
    mlflow.log_params({f'model.{k}': v for k, v in params.items()})

    logger.info("Сохраняем модель")
    dump(model, MODEL_FILEPATH)

    if model_type == 'logreg':
        mlflow.sklearn.log_model(model, artifact_path='model')
    elif model_type in {'dt', 'rf'}:
        mlflow.sklearn.log_model(model, artifact_path='model')
    else:
        mlflow.xgboost.log_model(model, artifact_path='model')

    logger.info("Успешно!")


if __name__ == "__main__":
    train()
