import os
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import shutil

TARGET_COLUMNS = ['R', 'I', 'A', 'S', 'E', 'C']

def load_and_preprocess_data(file_path, embedding_column='pca_emb'):
    df = pd.read_csv(file_path)
    df['holland_types'] = df['holland_types'].apply(ast.literal_eval)
    riasec_df = df['holland_types'].apply(pd.Series)
    df = pd.concat([df.drop(columns=['holland_types']), riasec_df], axis=1)

    df[embedding_column] = df[embedding_column].apply(ast.literal_eval)
    X = np.array(df[embedding_column].tolist())
    y = df[TARGET_COLUMNS].values
    user_ids = df['user_id'].values

    return X, y, user_ids, df

def split_data(X, y, test_size=0.3, val_size=0.5, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_predictions(y_pred, user_ids, true_df, output_path):
    pred_df = pd.DataFrame(y_pred, columns=TARGET_COLUMNS)
    pred_df['user_id'] = user_ids

    def to_dict_row(row):
        return {col: row[col] for col in TARGET_COLUMNS}

    pred_df['holland_types_pred'] = pred_df.apply(to_dict_row, axis=1)
    true_df = true_df[['user_id'] + TARGET_COLUMNS].copy()
    true_df['holland_types_true'] = true_df.apply(to_dict_row, axis=1)

    final_df = pd.merge(
        true_df[['user_id', 'holland_types_true']],
        pred_df[['user_id', 'holland_types_pred']],
        on='user_id'
    )
    final_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

def archive_output(output_dir, archive_name):
    shutil.make_archive(archive_name, 'zip', output_dir)
    print(f"Archived output to {archive_name}.zip")
