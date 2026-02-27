import os
import mlflow
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from constants import DATASET_NAME, DATASET_PATH_PATTERN, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'process_data'


def process_data():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали скачивать данные')
    dataset = load_dataset(DATASET_NAME)
    logger.info('Успешно скачали данные!')

    logger.info('Делаем предобработку данных')
    df = dataset['train'].to_pandas()
    columns = params['features']
    target_column = params['target_column']
    X, y = df[columns], df[target_column]
    logger.info(f'    Используемые фичи: {columns}')

    all_cat_features = [
        'workclass', 'education', 'marital.status', 'occupation', 'relationship',
        'race', 'sex', 'native.country',
    ]
    cat_features = list(set(columns) & set(all_cat_features))
    num_features = list(set(columns) - set(all_cat_features))

    preprocessor = OrdinalEncoder()
    # preprocessor = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_transformed = np.hstack([X[num_features], preprocessor.fit_transform(X[cat_features])])
    y_transformed: pd.Series = (y == '>50K').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_transformed, test_size=params['test_size'], random_state=RANDOM_STATE
    )

    train_size_param = params['train_size']
    X_train = X_train[:train_size_param]
    y_train = y_train.iloc[:train_size_param]

    mlflow.log_param("features", columns)
    mlflow.log_param("train_rows", len(y_train))
    mlflow.log_param("test_rows", len(y_test))
    mlflow.log_param("test_size", params['test_size']) 
    mlflow.log_param("random_state", RANDOM_STATE)

    logger.info(f'    Размер тренировочного датасета: {len(y_train)}')
    logger.info(f'    Размер тестового датасета: {len(y_test)}')

    logger.info('Начали сохранять датасеты')
    os.makedirs(os.path.dirname(DATASET_PATH_PATTERN), exist_ok=True)
    for split, split_name in zip(
        (X_train, X_test, y_train, y_test),
        ('X_train', 'X_test', 'y_train', 'y_test'),
    ):
        pd.DataFrame(split).to_csv(
            DATASET_PATH_PATTERN.format(split_name=split_name), index=False
        )
    logger.info('Успешно сохранили датасеты!')

    mlflow.log_artifact(DATASET_PATH_PATTERN.format(split_name="X_train"), artifact_path="dataset")
    mlflow.log_artifact(DATASET_PATH_PATTERN.format(split_name="X_test"), artifact_path="dataset")
    mlflow.log_artifact(DATASET_PATH_PATTERN.format(split_name="y_train"), artifact_path="dataset")
    mlflow.log_artifact(DATASET_PATH_PATTERN.format(split_name="y_test"), artifact_path="dataset")


if __name__ == '__main__':
    process_data()
