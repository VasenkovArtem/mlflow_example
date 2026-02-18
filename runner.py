import mlflow
from scripts import evaluate, process_data, train


if __name__ == '__main__':
    mlflow.set_tracking_uri('http://158.160.2.37:5000/')
    mlflow.set_experiment('homework_radilkhanova')

    with mlflow.start_run():
        process_data()
        train()
        evaluate()