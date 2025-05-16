import numpy as np
import optuna
from optuna.integration.botorch import BoTorchSampler
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from utils import load_and_preprocess_data, split_data, save_predictions, TARGET_COLUMNS

def objective(trial, X_train, y_train, X_val, y_val):
    param = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 1, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "loss_function": "MultiRMSE",
        "verbose": False
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    model = CatBoostRegressor(**param)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

def train_and_predict(file_path, output_path):
    X, y, user_ids, df = load_and_preprocess_data(file_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    study = optuna.create_study(direction='minimize', sampler=BoTorchSampler())
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=2, timeout=90.0)

    best_params = study.best_params
    print(f"Best CatBoost params for {file_path}: {best_params}")

    model = CatBoostRegressor(**best_params, loss_function="MultiRMSE")
    model.fit(X_train, y_train, verbose=100)

    y_pred = model.predict(X_test)

    save_predictions(y_pred, user_ids, df, output_path)

    return y_pred
