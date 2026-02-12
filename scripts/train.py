import os
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from constants import DATASET_PATH_PATTERN, RANDOM_STATE, RUN_ID_FILE
from utils import get_logger, load_params, generate_run_hash

STAGE_NAME = 'train'

def get_model(model_type, model_params):
    model_params['random_state'] = RANDOM_STATE
    if model_type == "LogisticRegression":
        return LogisticRegression(**model_params)
    elif model_type == "RandomForest":
        return RandomForestClassifier(**model_params)
    elif model_type == "GradientBoosting":
        return GradientBoostingClassifier(**model_params)
    elif model_type == "DecisionTreeClassifier":
        return DecisionTreeClassifier(**model_params)
    else:
        raise ValueError(f"unknown model_type: {model_type}")

def train():
    logger = get_logger(logger_name=STAGE_NAME)
    train_params = load_params("train")
    process_params = load_params("process_data")

    logger.info('Начали считывать датасеты')
    splits = []
    for split_name in ['X_train', 'X_test', 'y_train', 'y_test']:
        splits.append(pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name)))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    model_type = train_params['model_type']
    model_params = train_params['params'].copy()
    model_params['random_state'] = RANDOM_STATE
    logger.info(f'Тип модели: {model_type}, параметры: {model_params}')
    model = get_model(model_type, model_params)

    logger.info('Обучаем модель')
    model.fit(X_train, y_train)

    run_hash = os.environ.get("MLFLOW_RUN_HASH", "default")
    run_name = f"{model_type}_{run_hash}"

    mlflow.set_experiment("homework_ulyanova")
    with mlflow.start_run(run_name=run_name) as run:

        mlflow.log_param("train_size", process_params['params'].get("train_size"))
        mlflow.log_param("features", ",".join(process_params['params']["features"]))
        mlflow.log_param("run_hash", run_hash)

        mlflow.log_param("model_type", model_type)
        actual_model_params = model.get_params()
        for k, v in actual_model_params.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                mlflow.log_param(k, v)
            else:
                try:
                    mlflow.log_param(k, str(v))
                except Exception:
                    mlflow.log_param(k, "unserializable")

        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        with open(RUN_ID_FILE, "w") as f:
            f.write(run_id)
        logger.info(f"Saved run_id to {RUN_ID_FILE}: {run_id}")

    logger.info(f'Завершено обучение. run_id: {run_id}')