from scripts import evaluate, process_data, train, get_run


if __name__ == '__main__':
    mlflow, run = get_run.get_run()
    process_data(mlflow, run)
    train(mlflow, run)
    evaluate(mlflow, run)
