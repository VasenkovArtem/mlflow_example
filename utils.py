import logging
import os
import warnings

import mlflow
import yaml
from sklearn.exceptions import DataConversionWarning

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(name)s : %(message)s')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)

PARAMS_FILEPATH_PATTERN = '/app/params/{stage_name}.yaml'
MLFLOW_TRACKING_URI = 'http://158.160.2.37:5000'
MLFLOW_EXPERIMENT_NAME = 'homework_Bazhenov'


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


def init_mlflow(experiment_name: str | None = None) -> None:
    """
    Initialize MLflow tracking and experiment.

    Should be called once per pipeline run before starting mlflow.start_run().
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name or MLFLOW_EXPERIMENT_NAME)
