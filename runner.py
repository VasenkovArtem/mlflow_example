import mlflow
from datetime import datetime
from scripts import evaluate, process_data, train

def main():
    mlflow.set_tracking_uri('http://158.160.2.37:5000/')
    mlflow.set_experiment('homework_Pozdeeva')

    run_name = "run_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with mlflow.start_run(run_name=run_name):
        process_data()
        train()
        evaluate()

if __name__ == '__main__':
    main()
