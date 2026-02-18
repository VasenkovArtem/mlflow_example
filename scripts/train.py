import pandas as pd
from joblib import dump
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'


def train(mlflow, run):
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    logger.info('Создаём модель')
    params['random_state'] = RANDOM_STATE
    logger.info(f'    Параметры модели: {params}')
#     model = LogisticRegression(**params)
    model = RandomForestClassifier(**params)
#     model = XGBClassifier(**params)
    model_name = model.__class__.__name__
    
#     параметры млфлов
    mlflow.log_params({'model_type': str(model_name)}, run_id=run.info.run_id)
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.sklearn.log_model(model, 'model')
    mlflow.log_params(params, run_id=run.info.run_id)
    
    logger.info('Обучаем модель')
    model.fit(X_train, y_train)
    
#     coefficients = model.coef_[0]
#     logger.info(coefficients)
#     параметры млфлов
#     mlflow.log_param({'feature_importances': str(coefficients.tolist())}, run_id=run.info.run_id)

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)
    logger.info('Успешно!')


if __name__ == '__main__':
    train()
