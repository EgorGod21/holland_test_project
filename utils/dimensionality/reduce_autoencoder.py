import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self, input_dim, compressed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, compressed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def reduce_embeddings_autoencoder(folder_path: str, output_path: str, threshold: float = 0.95,
                                   batch_size: int = 64, lr: float = 1e-3, epochs: int = 1000,
                                   patience: int = 10):
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file in tqdm(files, desc="Autoencoder Reduction"):
        df = pd.read_csv(os.path.join(folder_path, file))
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(x if isinstance(x, list) else eval(x)))
        embedding_matrix = np.vstack(df['embeddings'].values)

        from sklearn.decomposition import PCA
        explained_variance = np.cumsum(PCA().fit(embedding_matrix).explained_variance_ratio_)
        compressed_dim = np.argmax(explained_variance >= threshold) + 1
        input_dim = embedding_matrix.shape[1]

        X_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = Autoencoder(input_dim, compressed_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        best_loss = float('inf')
        patience_counter = 0
        train_rmses = []

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch)
                loss = loss_fn(output, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_mse = epoch_loss / len(loader)
            rmse = avg_mse ** 0.5
            train_rmses.append(rmse)

            if best_loss - avg_mse > 1e-5:
                best_loss = avg_mse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        model.eval()
        with torch.no_grad():
            compressed_embeddings = model.encoder(X_tensor.to(device)).cpu().numpy()

        plt.figure(figsize=(7, 5))
        plt.plot(train_rmses, label='RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title(f'Training RMSE - {file}')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{file}_training_curve.png"))
        plt.close()

        df_out = pd.DataFrame({
            "user_id": df["user_id"],
            "pca_emb": [vec.tolist() for vec in compressed_embeddings]
        })
        df_out.to_csv(os.path.join(output_path, f"autoencoded_{file}"), index=False)
