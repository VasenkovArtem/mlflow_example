import os
_DATA_DIR = '/app/data' if os.path.exists('/app') else os.path.join(os.path.dirname(__file__), 'data')
DATASET_PATH_PATTERN = os.path.join(_DATA_DIR, '{split_name}.csv')
DATASET_NAME = 'scikit-learn/adult-census-income'
_MODEL_DIR = '/app' if os.path.exists('/app') else os.path.dirname(__file__)
MODEL_FILEPATH = os.path.join(_MODEL_DIR, 'model.joblib')
RANDOM_STATE = 42
TEST_SIZE = 0.3
MLFLOW_TRACKING_URI = 'http://158.160.2.37:5000/'
EXPERIMENT_NAME = 'homework_kislitsyna'