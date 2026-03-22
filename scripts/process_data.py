import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import mlflow

from constants import (
    DATASET_NAME,
    DATASET_PATH_PATTERN,
    RANDOM_STATE,
    TEST_DATASET_CSV_PATH,
    TEST_SIZE,
    TRAIN_DATASET_CSV_PATH,
)
from utils import get_logger, load_params

STAGE_NAME = 'process_data'

ALL_CAT_FEATURES = frozenset(
    (
        'workclass',
        'education',
        'marital.status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native.country',
    )
)


def _feature_column_order(columns: list[str]) -> tuple[list[str], list[str]]:
    num = [c for c in columns if c not in ALL_CAT_FEATURES]
    cat = [c for c in columns if c in ALL_CAT_FEATURES]
    return num, cat


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

    num_features, cat_features = _feature_column_order(columns)
    feature_names_after_transform = num_features + cat_features

    preprocessor = OrdinalEncoder()
    X_transformed = np.hstack(
        [X[num_features].to_numpy(), preprocessor.fit_transform(X[cat_features])]
    )
    y_transformed: pd.Series = (y == '>50K').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_transformed, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    if 'train_size' in params and params['train_size'] is not None:
        train_size = params['train_size']
        X_train = X_train[:train_size]
        y_train = y_train[:train_size]
        logger.info(f'    Урезали тренировочный датасет до {train_size} записей')

    logger.info(f'    Размер тренировочного датасета: {len(y_train)}')
    logger.info(f'    Размер тестового датасета: {len(y_test)}')

    mlflow.log_param('features', ','.join(sorted(columns)))
    mlflow.log_param('n_features', len(columns))
    mlflow.log_param('train_size', len(y_train))
    mlflow.log_param('test_size', len(y_test))
    mlflow.log_param('random_state', RANDOM_STATE)
    mlflow.log_param('test_split_ratio', TEST_SIZE)

    logger.info('Начали сохранять датасеты')
    os.makedirs(os.path.dirname(DATASET_PATH_PATTERN), exist_ok=True)
    for split, split_name in zip(
        (X_train, X_test, y_train, y_test),
        ('X_train', 'X_test', 'y_train', 'y_test'),
    ):
        if split_name.startswith('X_'):
            out = pd.DataFrame(split, columns=feature_names_after_transform)
        else:
            out = pd.DataFrame({'target': np.asarray(split).ravel()})
        out.to_csv(DATASET_PATH_PATTERN.format(split_name=split_name), index=False)
    logger.info('Успешно сохранили датасеты!')

    if params.get('log_dataset', True):
        logger.info('Логируем тренировочный датасет в MLflow')
        train_df = pd.DataFrame(X_train, columns=feature_names_after_transform)
        train_df['target'] = np.asarray(y_train).ravel()
        train_df.to_csv(TRAIN_DATASET_CSV_PATH, index=False)
        mlflow.log_artifact(TRAIN_DATASET_CSV_PATH, 'datasets')
        logger.info('Тренировочный датасет залогирован в MLflow!')

    # по вайбу
    if params.get('log_test_dataset', False):
        logger.info('Логируем тестовый датасет в MLflow')
        test_df = pd.DataFrame(X_test, columns=feature_names_after_transform)
        test_df['target'] = np.asarray(y_test).ravel()
        test_df.to_csv(TEST_DATASET_CSV_PATH, index=False)
        mlflow.log_artifact(TEST_DATASET_CSV_PATH, 'datasets')
        logger.info('Тестовый датасет залогирован в MLflow!')


if __name__ == '__main__':
    process_data()
