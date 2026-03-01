from scripts import evaluate, process_data, train
import mlflow

mlflow.set_tracking_uri("http://158.160.2.37:5000/")
mlflow.set_experiment("homework_klochkov")

if __name__ == '__main__':
    with mlflow.start_run(run_name="ohe_all_features_random_forest_data_save"):
        process_data()
        train()
        evaluate()

# if __name__ == '__main__':
#     process_data()
#     train()
#     evaluate()
