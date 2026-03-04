import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
import mlflow.catboost
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, get_scorer

from constants import DATASET_PATH_PATTERN, RANDOM_STATE
from setup import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, log_mlflow
from utils import get_logger, load_params

STAGE_NAME = 'optimize'
logger = get_logger(logger_name=STAGE_NAME)


def objective(trial):
    X_train = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='X_train'))
    y_train = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='y_train')).values.ravel()
    X_test = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='X_test'))
    y_test = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='y_test')).values.ravel()
    
    params = {
        'iterations': trial.suggest_int('iterations', 50, 500, step=50),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'thread_count': -1
    }
    
    try:
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        scores = {
            f"metric_{metric_name}": get_scorer(metric_name)(model, X_test, y_test) 
            for metric_name in load_params(stage_name='evaluate')['metrics']
        }
        log_mlflow(**scores)
        
        return roc_auc
    
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        return 0.0


def optimize():
    mlflow.end_run()
    params = load_params(stage_name=STAGE_NAME)
    n_trials = params.get('n_trials', 50)
    
    mlflc = MLflowCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        metric_name="roc_auc",
        create_experiment=False,
        mlflow_kwargs={
            "experiment_name": EXPERIMENT_NAME,
            "nested": True 
        }
    )
    
    study = optuna.create_study(
        direction='maximize',
        study_name='catboost_hyperopt',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    
    logger.info(f'Запускаем {n_trials} trials с CatBoost...')
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[mlflc],
        show_progress_bar=True,
        n_jobs=1
    )
    
    logger.info(f'Лучший ROC-AUC: {study.best_value:.4f}')
    logger.info(f'Лучшие параметры:')
    for param_name, param_value in study.best_params.items():
        logger.info(f'  {param_name}: {param_value}')
    
    mlflow.end_run()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="optuna_best_catboost"):
        log_mlflow(
            param_model_type="CatBoost",
            **{f"param_{k}": v for k, v in study.best_params.items()}
        )
        
        log_mlflow(
            metric_best_roc_auc=study.best_value,
            metric_n_trials=len(study.trials)
        )
        
        logger.info('Обучаем финальную модель с лучшими параметрами...')
        X_train = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='X_train'))
        y_train = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='y_train')).values.ravel()
        X_test = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='X_test'))
        y_test = pd.read_csv(DATASET_PATH_PATTERN.format(split_name='y_test')).values.ravel()
        
        best_model = CatBoostClassifier(
            **study.best_params,
            random_seed=RANDOM_STATE,
            verbose=False
        )
        best_model.fit(X_train, y_train, verbose=False)
        
        mlflow.catboost.log_model(best_model, artifact_path="model", input_example=X_train[:5])
        
        y_proba = best_model.predict_proba(X_test)[:, 1]
        final_roc_auc = roc_auc_score(y_test, y_proba)
        scores = {
            f"metric_best_{metric_name}": get_scorer(metric_name)(best_model, X_test, y_test) 
            for metric_name in load_params(stage_name='evaluate')['metrics']
        }
        log_mlflow(**scores)
        
        trials_df = study.trials_dataframe()
        trials_df.to_csv("optuna_trials.csv", index=False)
        mlflow.log_artifact("optuna_trials.csv")
        
        logger.info(f'Финальный ROC-AUC лучшей модели: {final_roc_auc:.4f}')
    
    logger.info('Все результаты залогированы в MLflow!')


if __name__ == '__main__':
    optimize()