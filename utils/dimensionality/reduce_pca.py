import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

def reduce_embeddings_pca(folder_path: str, output_path: str, threshold: float = 0.95):
    os.makedirs(output_path, exist_ok=True)
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file in tqdm(files, desc="PCA Reduction"):
        df = pd.read_csv(os.path.join(folder_path, file))
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(x if isinstance(x, list) else eval(x)))
        embedding_matrix = np.vstack(df['embeddings'].values)

        pca = PCA().fit(embedding_matrix)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        compressed_dim = np.argmax(explained_variance >= threshold) + 1

        pca = PCA(n_components=compressed_dim)
        reduced_embeddings = pca.fit_transform(embedding_matrix)

        df_out = pd.DataFrame({
            "user_id": df["user_id"],
            "pca_emb": [vec.tolist() for vec in reduced_embeddings]
        })

        df_out.to_csv(os.path.join(output_path, f"pca_{file}"), index=False)
