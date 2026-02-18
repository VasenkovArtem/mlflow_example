import os
import mlflow
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from constants import DATASET_NAME, DATASET_PATH_PATTERN, TEST_SIZE, RANDOM_STATE
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
    train_size = params.get('train_size', None)

    target_column = 'income'
    X, y = df[columns], df[target_column]

    logger.info(f'    Используемые фичи: {columns}')
    mlflow.log_param("features", ",".join(columns))


    # список всех категориальных признаков
    all_cat_features = [
        'workclass', 'education', 'marital.status', 'occupation',
        'relationship', 'race', 'sex', 'native.country',
    ]

    cat_features = list(set(columns) & set(all_cat_features))
    num_features = list(set(columns) - set(all_cat_features))

    logger.info(f'    Категориальные признаки: {cat_features}')
    logger.info(f'    Числовые признаки: {num_features}')

    # кодирование категориальных признаков
    preprocessor = OrdinalEncoder()

    X_cat = (
        preprocessor.fit_transform(X[cat_features])
        if len(cat_features) > 0
        else np.empty((len(X), 0))
    )

    X_num = (
        X[num_features].to_numpy()
        if len(num_features) > 0
        else np.empty((len(X), 0))
    )

    X_transformed = np.hstack([X_num, X_cat])

    # бинаризация таргета
    y_transformed: pd.Series = (y == '>50K').astype(int)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed,
        y_transformed,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # =========================
    # использование train_size
    # =========================
    if train_size is not None:
        logger.info(f'    Обрезаем train dataset до {train_size} объектов')
        train_size = min(train_size, len(X_train))
        X_train = X_train[:train_size]
        y_train = y_train.iloc[:train_size]
        mlflow.log_param("train_size", train_size)
    else:
        mlflow.log_param("train_size", "full")

    logger.info(f'    Размер тренировочного датасета: {len(y_train)}')
    logger.info(f'    Размер тестового датасета: {len(y_test)}')

    # =========================
    # сохранение датасетов
    # =========================
    logger.info('Начали сохранять датасеты')

    os.makedirs(os.path.dirname(DATASET_PATH_PATTERN), exist_ok=True)

    for split, split_name in zip(
        (X_train, X_test, y_train, y_test),
        ('X_train', 'X_test', 'y_train', 'y_test'),
    ):
        pd.DataFrame(split).to_csv(
            DATASET_PATH_PATTERN.format(split_name=split_name),
            index=False,
        )

    logger.info('Успешно сохранили датасеты!')
    dataset_dir = os.path.dirname(DATASET_PATH_PATTERN)
    mlflow.log_artifacts(
        dataset_dir,
        artifact_path="dataset"
    )

if __name__ == '__main__':
    process_data()
