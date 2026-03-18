import mlflow
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'

MODEL_BUILDERS = {
    'logistic_regression': LogisticRegression,
    'decision_tree': DecisionTreeClassifier,
    'random_forest': RandomForestClassifier,
    'gradient_boosting': GradientBoostingClassifier,
    'xgboost': XGBClassifier,
}


def _flatten_for_mlflow(prefix: str, cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        if v is None:
            continue
        key = f'{prefix}_{k}'[:250]
        val = str(v)
        if len(val) > 500:
            val = val[:497] + '...'
        out[key] = val
    return out


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    model_type = params['model_type']
    if model_type not in MODEL_BUILDERS:
        raise ValueError(
            f'Неизвестный model_type: {model_type}. Допустимо: {list(MODEL_BUILDERS)}'
        )

    model_cfg = dict(params.get(model_type) or {})
    logger.info('Создаём модель')
    logger.info(f'    model_type={model_type}, cfg={model_cfg}')

    cls = MODEL_BUILDERS[model_type]
    if model_type == 'xgboost':
        model = cls(random_state=RANDOM_STATE, **model_cfg)
    else:
        model = cls(random_state=RANDOM_STATE, **model_cfg)

    logger.info('Обучаем модель')
    y_tr = y_train.values.ravel() if hasattr(y_train, 'values') else y_train
    model.fit(X_train, y_tr)

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)

    if mlflow.active_run():
        mlflow.log_param('model_type', model_type)
        mlflow.log_params(_flatten_for_mlflow('model', model_cfg))
        input_example = X_train.iloc[:3]
        if model_type == 'xgboost':
            mlflow.xgboost.log_model(
                model, artifact_path='model', input_example=input_example
            )
        else:
            mlflow.sklearn.log_model(
                model, artifact_path='model', input_example=input_example
            )

    logger.info('Успешно!')


if __name__ == '__main__':
    train()
