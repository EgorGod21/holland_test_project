import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def evaluate_models(base_dir='../data/'):
    folders = [f for f in os.listdir(base_dir)
              if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('output')]

    result = {}

    for folder in tqdm(folders):
        path = base_dir + folder
        folder_paths = [path + "/pca_emb/", path + "/autoencoder_emb/"]
        folder_paths_emb = ["pca_emb", "autoencoder_emb"]
        model = folder.split('_')[1]
        result[model] = {}

        for folder_path, folder_path_emb in zip(folder_paths, folder_paths_emb):
            files = [
                f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".csv")
            ]
            result[model][folder_path_emb] = {}

            for file in files:
                data = pd.read_csv(folder_path + file, sep=',')
                model_emb = '_'.join(os.path.splitext(file)[0].rsplit('_', 2)[-2:])

                data['holland_types_pred'] = data['holland_types_pred'].apply(eval)
                data['holland_types_true'] = data['holland_types_true'].apply(eval)

                true_df = data['holland_types_true'].apply(pd.Series)
                pred_df = data['holland_types_pred'].apply(pd.Series)

                rmse_total = np.sqrt(mean_squared_error(true_df, pred_df))
                rmse_per_column = np.sqrt(np.mean((true_df - pred_df) ** 2, axis=0))

                # Calculate top3 matches
                data['top3_predicted_types'] = None
                data['top3_actual_types'] = None
                data['number_of_matching_types'] = 0
                data['matching_types'] = None

                for index, row in data.iterrows():
                    predicted_types = row['holland_types_pred']
                    actual_types = row['holland_types_true']

                    top3_predicted = dict(sorted(predicted_types.items(), key=lambda item: item[1], reverse=True)[:3])
                    top3_actual = dict(sorted(actual_types.items(), key=lambda item: item[1], reverse=True)[:3])

                    predicted_keys = set(top3_predicted.keys())
                    actual_keys = set(top3_actual.keys())
                    intersection = predicted_keys.intersection(actual_keys)

                    data.at[index, 'top3_predicted_types'] = top3_predicted
                    data.at[index, 'top3_actual_types'] = top3_actual
                    data.at[index, 'number_of_matching_types'] = len(intersection)
                    data.at[index, 'matching_types'] = list(intersection)

                matches = data.number_of_matching_types.value_counts().to_dict()
                result[model][folder_path_emb][model_emb] = {
                    'RMSE_total': float(rmse_total),
                    'RMSE_per': {col: float(val) for col, val in rmse_per_column.items()},
                    'top3_predicted_types': [
                        {k: float(v) for k, v in d.items()} for d in data.top3_predicted_types.tolist()
                    ],
                    'top3_actual_types': [
                        {k: float(v) for k, v in d.items()} for d in data.top3_actual_types.tolist()
                    ],
                    'number_of_matching_types': {
                        int(k): int(v) for k, v in data.number_of_matching_types.value_counts().to_dict().items()
                    }
                }

    return result
