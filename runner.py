import os
import mlflow
from utils import load_params, generate_run_hash
from scripts import process_data, train, evaluate

if __name__ == '__main__':
    
    mlflow.set_tracking_uri("http://158.160.2.37:5000")
    
    train_params = load_params("train")
    process_params = load_params("process_data")
    
    run_hash = generate_run_hash(train_params, process_params)
    os.environ["MLFLOW_RUN_HASH"] = run_hash
    
    print(f"MLflow tracking url: {mlflow.get_tracking_uri()}")
    print(f"Starting pipeline with run_hash: {run_hash}")
    print(f"Starting pipeline with train_params: {train_params}")
    print(f"Starting pipeline with process_params: {process_params}")
    
    process_data()
    train()
    evaluate()