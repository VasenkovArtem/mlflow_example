import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from constants import DATASET_NAME, DATASET_PATH_PATTERN, TEST_SIZE, RANDOM_STATE
from utils import get_logger, load_params

import mlflow
from setup import log_mlflow

STAGE_NAME = 'process_data'


def process_data():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали скачивать данные')
    dataset = load_dataset(DATASET_NAME)
    logger.info('Успешно скачали данные!')

    logger.info('Делаем предобработку данных')
    df = dataset['train'].to_pandas()
    columns = sorted(params['features'])
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

    train_size = params.get('train_size', len(X_train))
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    
    log_mlflow(param_features=",".join(columns), 
               param_n_features=len(columns), 
               param_test_size_fraction=TEST_SIZE,
               param_train_samples=len(y_train),
               param_test_samples=len(y_test))

    logger.info('Логируем датасеты в MLflow как артефакты...')
    
    train_df = pd.DataFrame(X_train)
    train_df['target'] = y_train.values
    train_df.to_csv("train_dataset.csv", index=False)
    mlflow.log_artifact("train_dataset.csv")
    
    test_df = pd.DataFrame(X_test)
    test_df['target'] = y_test.values
    test_df.to_csv("test_dataset.csv", index=False)
    mlflow.log_artifact("test_dataset.csv")
    
    feature_info = {
        'features': columns,
        'n_features': len(columns),
        'categorical_features': cat_features,
        'numerical_features': num_features,
        'train_size': len(y_train),
        'test_size': len(y_test)
    }
    import json
    with open("dataset_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)
    mlflow.log_artifact("dataset_info.json")
    
    logger.info('Датасеты залогированы в MLflow!')
    logger.info('Этап обработки данных успешно залогирован в MLflow')
    
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


if __name__ == '__main__':
    process_data()
