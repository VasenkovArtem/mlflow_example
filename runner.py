import mlflow

from scripts import evaluate, process_data, train


if __name__ == '__main__':
    mlflow.set_tracking_uri('http://158.160.2.37:5000/')
    mlflow.set_experiment("homework_pak")

    with mlflow.start_run(run_name='xgb_400_10_good_feat_22000'):
        process_data()
        train()
        evaluate()