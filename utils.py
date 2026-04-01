import logging
import os
import yaml
import warnings

import json
import mlflow

from sklearn.exceptions import DataConversionWarning
from constants import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(name)s : %(message)s')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)

PARAMS_FILEPATH_PATTERN = '/app/params/{stage_name}.yaml'


def load_params(stage_name: str) -> dict:
    params_filepath = PARAMS_FILEPATH_PATTERN.format(stage_name=stage_name)
    if not os.path.exists(params_filepath):
        raise FileNotFoundError(
            f'Параметров для шага {stage_name} не существует! Проверьте имя шага'
        )
    with open(params_filepath, 'r') as file:
        params = yaml.safe_load(file)
    return params['params']


def get_logger(
    logger_name: str | None = None,
    level: int = 20,
) -> logging.Logger:
    logger = logging.getLogger(name=logger_name)
    logger.setLevel(level)
    return logger


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def prepare_params_for_logging(params: dict, prefix: str = '') -> dict:
    logged = {}
    for key, value in params.items():
        log_key = f'{prefix}{key}'
        if isinstance(value, (list, tuple, dict, set)):
            logged[log_key] = json.dumps(value, ensure_ascii=False)
        else:
            logged[log_key] = value
    return logged
