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
    cat_features = [c for c in columns if c in all_cat_features]
    num_features = [c for c in columns if c not in all_cat_features]

    if cat_features and num_features:
        preprocessor = OrdinalEncoder()
        X_cat = preprocessor.fit_transform(X[cat_features])
        X_num = X[num_features].to_numpy(dtype=np.float64)
        X_transformed = np.hstack([X_num, X_cat])
    elif cat_features:
        preprocessor = OrdinalEncoder()
        X_transformed = preprocessor.fit_transform(X[cat_features])
    else:
        X_transformed = X[num_features].to_numpy(dtype=np.float64)

    y_transformed: pd.Series = (y == '>50K').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed,
        y_transformed,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_size = int(params['train_size'])
    n = min(train_size, len(y_train))
    X_train = X_train[:n]
    y_train = y_train.iloc[:n].reset_index(drop=True)

    logger.info(f'    Размер тренировочного датасета: {len(y_train)}')
    logger.info(f'    Размер тестового датасета: {len(y_test)}')

    if mlflow.active_run():
        mlflow.log_param('data_train_size_limit', train_size)
        mlflow.log_param('data_actual_train_rows', len(y_train))
        mlflow.log_param('data_test_rows', len(y_test))
        mlflow.log_param('data_test_size_ratio', TEST_SIZE)
        mlflow.log_param('data_feature_count', len(columns))
        feats_joined = ','.join(columns)
        if len(feats_joined) <= 500:
            mlflow.log_param('data_features', feats_joined)
        else:
            mlflow.log_param('data_features_prefix', feats_joined[:497] + '...')

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

    # Продвинутый уровень: артефакты обучающей выборки в MLflow (≥3 run с ними — при нескольких запусках runner)
    if mlflow.active_run():
        train_artifact_dir = 'train_dataset'
        for split_name in ('X_train', 'y_train'):
            path = DATASET_PATH_PATTERN.format(split_name=split_name)
            if os.path.isfile(path):
                mlflow.log_artifact(path, artifact_path=train_artifact_dir)
        logger.info(
            f'    Залогированы артефакты обучающего датасета в MLflow ({train_artifact_dir}/)'
        )


if __name__ == '__main__':
    process_data()
