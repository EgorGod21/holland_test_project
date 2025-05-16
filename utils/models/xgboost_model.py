import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from utils import load_and_preprocess_data, split_data, save_predictions, TARGET_COLUMNS

def objective(trial, X_train, y_train, X_val, y_val):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "alpha": trial.suggest_float("alpha", 0.0, 1.0),
        "lambda": trial.suggest_float("lambda", 0.0, 1.0),
    }

    model = XGBRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

def train_and_predict(file_path, output_path):
    X, y, user_ids, df = load_and_preprocess_data(file_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=3, timeout=90.0)

    best_params = study.best_params
    print(f"Best XGBoost params for {file_path}: {best_params}")

    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    save_predictions(y_pred, user_ids, df, output_path)

    return y_pred
