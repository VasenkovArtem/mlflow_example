import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from constants import DATASET_NAME, DATASET_PATH_PATTERN, TEST_SIZE, RANDOM_STATE
from utils import get_logger, load_params

import mlflow

STAGE_NAME = 'process_data'


def process_data(params_filepath: str | None = None):
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME, params_filepath=params_filepath)

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
        'workclass', 'education', 'marital.status', 'occupation', 'relationship',
        'race', 'sex', 'native.country',
    ]
    cat_features = list(set(columns) & set(all_cat_features))
    num_features = list(set(columns) - set(all_cat_features))

    preprocessor = OrdinalEncoder()
    X_transformed = np.hstack([X[num_features], preprocessor.fit_transform(X[cat_features])])
    y_transformed: pd.Series = (y == '>50K').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_transformed, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # use train_size param to take only train_size rows of train dataset
    X_train = X_train[:int(params['train_size'])]
    y_train = y_train[:int(params['train_size'])]

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

    data_processing_params = {
        'all_cat_features': all_cat_features,
        'cat_features': cat_features,
        'num_features': num_features,
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'train_size': params['train_size'],
    }

    for f_name, f_value in data_processing_params.items():
        mlflow.log_param(f_name, f_value)
    logger.info('Data processing parameters logged')

    # save dataset as artifact
     # Log training dataset as artifact to MLflow
    if params.get('save_dataset_as_artifact', True):
        logger.info('Create dataset artifact in MLflow')
        train_dataset = pd.concat([
            pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])]),
            pd.DataFrame(y_train, columns=['target'])
        ], axis=1)
        train_dataset_path = '/tmp/train_dataset.csv'
        train_dataset.to_csv(train_dataset_path, index=False)
        mlflow.log_artifact(train_dataset_path, 'datasets')
        logger.info('Dataset artifact created in§ MLflow!')


if __name__ == '__main__':
    process_data()
