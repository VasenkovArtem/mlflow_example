import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
import mlflow

from constants import DATASET_NAME, DATASET_PATH_PATTERN, TEST_SIZE, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'process_data'


def process_data():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)
    mlflow.log_params(params)

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
        'race', 'sex', 'native.country', 'hours.per.week'
    ]
    cat_features = list(set(columns) & set(all_cat_features))
    num_features = list(set(columns) - set(all_cat_features))

    if params["cat_encoder_type"] == "ohe":
        preprocessor = OneHotEncoder(sparse_output=False)
    else:
        preprocessor = OrdinalEncoder()
    ss = StandardScaler()
    X_transformed = np.hstack([ss.fit_transform(X[num_features]), preprocessor.fit_transform(X[cat_features])])
    y_transformed: pd.Series = (y == '>50K').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_transformed, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # use train_size param to take only train_size rows of train dataset
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, train_size=params['train_size'], random_state=RANDOM_STATE
    )

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
        mlflow.log_artifact(DATASET_PATH_PATTERN.format(split_name=split_name), artifact_path="datasets")
    logger.info('Успешно сохранили датасеты!')


if __name__ == '__main__':
    process_data()
