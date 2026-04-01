import os
import mlflow
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from constants import DATASET_NAME, DATASET_PATH_PATTERN, TEST_SIZE, RANDOM_STATE
from utils import get_logger, load_params, prepare_params_for_logging

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
    train_size = params.get('train_size')

    X = df[columns].copy()
    y = (df[target_column] == '>50K').astype(int)

    logger.info(f'Используемые фичи: {columns}')

    all_cat_features = [
        'workclass', 'education', 'marital.status', 'occupation', 'relationship',
        'race', 'sex', 'native.country',
    ]
    cat_features = list(set(columns) & set(all_cat_features))
    num_features = list(set(columns) - set(all_cat_features))

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    train_parts = []
    test_parts = []

    if num_features:
        train_parts.append(X_train_raw[num_features].reset_index(drop=True))
        test_parts.append(X_test_raw[num_features].reset_index(drop=True))

    if cat_features:
        X_train_cat = pd.DataFrame(
            encoder.fit_transform(X_train_raw[cat_features]),
            columns=cat_features,
            index=X_train_raw.index,
        ).reset_index(drop=True)

        X_test_cat = pd.DataFrame(
            encoder.transform(X_test_raw[cat_features]),
            columns=cat_features,
            index=X_test_raw.index,
        ).reset_index(drop=True)

        train_parts.append(X_train_cat)
        test_parts.append(X_test_cat)

    X_train = pd.concat(train_parts, axis=1)
    X_test = pd.concat(test_parts, axis=1)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    if train_size is not None:
        actual_train_size = min(train_size, len(X_train))
        X_train = X_train.iloc[:actual_train_size].copy()
        y_train = y_train.iloc[:actual_train_size].copy()
    else:
        actual_train_size = len(X_train)

    logger.info(f'Размер тренировочного датасета: {len(y_train)}')
    logger.info(f'Размер тестового датасета: {len(y_test)}')

    logger.info('Начали сохранять датасеты')
    os.makedirs(os.path.dirname(DATASET_PATH_PATTERN), exist_ok=True)

    split_to_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train.to_frame(name='target'),
        'y_test': y_test.to_frame(name='target'),
    }

    for split_name, split_data in split_to_data.items():
        split_data.to_csv(DATASET_PATH_PATTERN.format(split_name=split_name), index=False)

    logger.info('Успешно сохранили датасеты!')

    if mlflow.active_run():
        mlflow.log_params(prepare_params_for_logging({
            'features': columns,
            'cat_features': cat_features,
            'num_features': num_features,
            'train_size_requested': train_size,
            'train_size_actual': actual_train_size,
            'test_size_actual': len(y_test),
            'random_state': RANDOM_STATE,
            'test_share': TEST_SIZE,
        }, prefix='data_'))

        for split_name in split_to_data:
            mlflow.log_artifact(
                DATASET_PATH_PATTERN.format(split_name=split_name),
                artifact_path='datasets'
            )
