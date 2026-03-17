import os

import mlflow
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from constants import DATASET_NAME, DATASET_PATH_PATTERN, RANDOM_STATE, TEST_SIZE
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
    target_column = 'income'
    X, y = df[columns], df[target_column]
    logger.info(f'    Используемые фичи: {columns}')

    all_cat_features = [
        'workclass',
        'education',
        'marital.status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native.country',
    ]
    cat_features = list(set(columns) & set(all_cat_features))
    num_features = list(set(columns) - set(all_cat_features))

    preprocessor = OrdinalEncoder()
    X_transformed = np.hstack(
        [X[num_features], preprocessor.fit_transform(X[cat_features])]
    )
    y_transformed: pd.Series = (y == '>50K').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_transformed, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # use train_size param to take only train_size rows of train dataset
    train_size = params.get('train_size')
    if train_size is not None:
        X_train = X_train[:train_size]
        y_train = y_train[:train_size]

    logger.info(f'    Размер тренировочного датасета: {len(y_train)}')
    logger.info(f'    Размер тестового датасета: {len(y_test)}')

    # логирование параметров подготовки данных в MLflow (при активном run)
    if mlflow.active_run() is not None:
        mlflow.log_param('data_features', ','.join(columns))
        if train_size is not None:
            mlflow.log_param('data_train_size_param', train_size)
        mlflow.log_param('data_num_features', len(num_features))
        mlflow.log_param('data_cat_features', len(cat_features))
        mlflow.log_param('data_train_size_actual', int(len(y_train)))
        mlflow.log_param('data_test_size_actual', int(len(y_test)))

    logger.info('Начали сохранять датасеты')
    data_dir = os.path.dirname(DATASET_PATH_PATTERN)
    os.makedirs(data_dir, exist_ok=True)

    saved_paths = []
    for split, split_name in zip(
        (X_train, X_test, y_train, y_test),
        ('X_train', 'X_test', 'y_train', 'y_test'),
    ):
        split_path = DATASET_PATH_PATTERN.format(split_name=split_name)
        pd.DataFrame(split).to_csv(split_path, index=False)
        saved_paths.append(split_path)

    logger.info('Успешно сохранили датасеты!')

    # логирование финальных датасетов как артефактов MLflow
    if mlflow.active_run() is not None:
        for split_path in saved_paths:
            mlflow.log_artifact(split_path, artifact_path='datasets')


if __name__ == '__main__':
    process_data()
