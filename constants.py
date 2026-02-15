import os

DATASET_PATH_PATTERN = '/app/data/{split_name}.csv'
DATASET_NAME = 'scikit-learn/adult-census-income'
MODEL_FILEPATH = '/app/model.joblib'
RANDOM_STATE = 42
TEST_SIZE = 0.3
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ID_FILE = os.path.join(BASE_DIR, "run_id.txt")