from scripts import evaluate, process_data, train
import mlflow
import sys

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://158.160.2.37:5000/")
    mlflow.set_experiment("homework_nagovitsyn")

    run_name = sys.argv[1] if len(sys.argv) > 1 else "unnamed_run"
    
    with mlflow.start_run(run_name=run_name):
        process_data()
        train()
        evaluate()
