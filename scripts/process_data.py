import os
import numpy as np
import pandas as pd
import mlflow
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from constants import DATASET_NAME, DATASET_PATH_PATTERN, TEST_SIZE, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'process_data'


def process_data():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Скачивание данных')
    dataset = load_dataset(DATASET_NAME)
    logger.info('Данные загружены')

    df = dataset['train'].to_pandas()
    columns = params['features']
    target_column = 'income'
    X, y = df[columns], df[target_column]
    logger.info(f'Признаки: {columns}')

    all_cat_features = [
        'workclass', 'education', 'marital.status', 'occupation', 'relationship',
        'race', 'sex', 'native.country',
    ]
    cat_features = [c for c in columns if c in all_cat_features]
    num_features = [c for c in columns if c not in all_cat_features]

    preprocessor = OrdinalEncoder()
    preprocessor.fit(X[cat_features])
    X_cat = preprocessor.transform(X[cat_features]) if cat_features else np.empty((len(X), 0))
    cat_idx = {c: i for i, c in enumerate(cat_features)}
    parts = []
    for col in columns:
        if col in num_features:
            parts.append(X[[col]].values)
        else:
            parts.append(X_cat[:, [cat_idx[col]]])
    X_transformed = np.hstack(parts)
    y_transformed: pd.Series = (y == '>50K').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_transformed, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    train_size = params.get('train_size')
    if train_size is not None:
        train_size = min(int(train_size), len(X_train))
        X_train = X_train[:train_size]
        y_train = y_train[:train_size]

    logger.info(f'Размер train: {len(y_train)}, test: {len(y_test)}')

    if mlflow.active_run():
        log_params = {
            'data_train_size': len(y_train),
            'data_test_size': len(y_test),
            'data_features': ','.join(columns),
        }
        if 'features_set' in params:
            log_params['data_features_set'] = params['features_set']
        mlflow.log_params(log_params)

    logger.info('Сохранение датасетов')
    os.makedirs(os.path.dirname(DATASET_PATH_PATTERN), exist_ok=True)
    for split, split_name in zip(
        (X_train, X_test, y_train, y_test),
        ('X_train', 'X_test', 'y_train', 'y_test'),
    ):
        pd.DataFrame(split).to_csv(
            DATASET_PATH_PATTERN.format(split_name=split_name), index=False
        )
    logger.info('Датасеты сохранены')


if __name__ == '__main__':
    process_data()
