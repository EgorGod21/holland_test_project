import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .utils import load_and_preprocess_data, split_data, save_predictions, TARGET_COLUMNS

BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 768

class MalwareDetectionHybrid(nn.Module):
    def __init__(self):
        super(MalwareDetectionHybrid, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.rnn = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(64 * 2, 256)
        self.fc2 = nn.Linear(256, 6)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 768)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def predict_on_all(model, full_loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch in full_loader:
            X_batch = X_batch[0].to(DEVICE)
            outputs = model(X_batch)
            preds.append(outputs.cpu().numpy())
    return np.vstack(preds)

def train_and_predict(file_path, output_path):
    X, y, user_ids, df = load_and_preprocess_data(file_path)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_tensor, y_tensor)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    full_loader = DataLoader(TensorDataset(X_tensor), batch_size=BATCH_SIZE)

    model = MalwareDetectionHybrid().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Training Hybrid NN on file: {file_path}")
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)

    predictions = predict_on_all(model, full_loader)
    predictions_original_scale = scaler_y.inverse_transform(predictions)

    save_predictions(predictions_original_scale, user_ids, df, output_path)

    return predictions_original_scale
