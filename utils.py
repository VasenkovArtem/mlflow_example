import logging
import os
import yaml
import warnings
import hashlib
import json

from sklearn.exceptions import DataConversionWarning

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
    return params


def get_logger(
    logger_name: str | None = None,
    level: int = 20,
) -> logging.Logger:
    logger = logging.getLogger(name=logger_name)
    logger.setLevel(level)
    return logger


def generate_run_hash(train_params, process_params):
    """Создаёт детерминированный хеш на основе конфигов при обучении модели.
    Хэш будет использован как уникальный идентификатор запуска.
    Используется для train и val, чтобы работать с одной и той же версией модели."""
    
    config = {
        "model_type": train_params["model_type"],
        "model_params": train_params["params"],
        "features": sorted(process_params["params"]["features"]),
        "train_size": process_params.get("train_size"),
    }
    config_str = json.dumps(config, sort_keys=True, default=str)
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()[:12]