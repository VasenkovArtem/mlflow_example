import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    logger.info('Создаём модель')
    
    # Extract model type from params
    model_type = params.get('model_type', 'LogisticRegression')
    model_params = {k: v for k, v in params.items() if k != 'model_type'}
    model_params['random_state'] = RANDOM_STATE
    
    # Model factory
    model_classes = {
        'LogisticRegression': LogisticRegression,
        'DecisionTree': DecisionTreeClassifier,
        'RandomForest': RandomForestClassifier,
        'XGBoost': XGBClassifier,
    }
    
    if model_type not in model_classes:
        raise ValueError(f'Unknown model type: {model_type}. Available: {list(model_classes.keys())}')
    
    logger.info(f'    Тип модели: {model_type}')
    logger.info(f'    Параметры модели: {model_params}')
    
    model = model_classes[model_type](**model_params)
    
    # Log model parameters to MLflow
    mlflow.log_param('model_type', model_type)
    mlflow.log_params(model_params)

    logger.info('Обучаем модель')
    model.fit(X_train, y_train.values.ravel())

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)
    
    # Log model to MLflow
    mlflow.sklearn.log_model(model, 'model')
    
    # Log feature importances if available
    if hasattr(model, 'feature_importances_'):
        import matplotlib.pyplot as plt
        import numpy as np
        
        feature_importances = model.feature_importances_
        indices = np.argsort(feature_importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(feature_importances)), feature_importances[indices])
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('/tmp/feature_importances.png')
        mlflow.log_artifact('/tmp/feature_importances.png')
        plt.close()
        
        logger.info('Залогировали feature importances')
    
    logger.info('Успешно!')


if __name__ == '__main__':
    train()
