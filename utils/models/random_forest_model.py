# scripts/random_forest_model.py
import numpy as np
import optuna
from optuna.integration.botorch import BoTorchSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from utils import load_and_preprocess_data, split_data, save_predictions, TARGET_COLUMNS

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 100, 500, step=100),
        'max_depth': trial.suggest_int("max_depth", 10, 110, step=20),
        'min_samples_split': trial.suggest_int("min_samples_split", 2, 10, step=2),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 4, step=1),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

def train_and_predict(file_path, output_path):
    X, y, user_ids, df = load_and_preprocess_data(file_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    study = optuna.create_study(direction='minimize', sampler=BoTorchSampler())
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=1, timeout=90.0)

    best_params = study.best_params
    print(f"Best Random Forest params for {file_path}: {best_params}")

    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    save_predictions(y_pred, user_ids[:len(y_pred)], df, output_path)

    return y_pred
