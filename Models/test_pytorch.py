import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess test data
file_path = '/Users/mchildress/Code/dreamers/synthetic_time_series.csv'
data = pd.read_csv(file_path)
data = data.sort_values(by='x')

# Calculate the change from the previous step
data['y_diff'] = data['y'].diff()
data['y_binary'] = (data['y_diff'] > 0).astype(int)
data = data.dropna()

# Feature engineering
data['lag1'] = data['y'].shift(1)
data['lag2'] = data['y'].shift(2)
data['lag3'] = data['y'].shift(3)
data['rolling_mean_3'] = data['y'].rolling(window=3).mean()
data['rolling_std_3'] = data['y'].rolling(window=3).std()
data['ema_3'] = data['y'].ewm(span=3, adjust=False).mean()
data = data.dropna()

# Extract features and target
X = data.drop(['y', 'y_diff', 'y_binary'], axis=1)
y = data['y_binary']

# Standardize features using the previously saved scaler
scaler = joblib.load("scaler.pkl")
X_scaled = scaler.transform(X)

# Convert to PyTorch tensors
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Define the model


class NeuralNet(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5, l2_reg=0.01):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.l2_reg = l2_reg

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Load the best model
input_size = X.shape[1]
model = NeuralNet(input_size, dropout_rate=0.3, l2_reg=0.001).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Predict probabilities
probabilities = []
with torch.no_grad():
    for X_batch, _ in data_loader:
        outputs = model(X_batch).squeeze()
        probabilities.extend(outputs.cpu().numpy())

# Adjust threshold for predictions
threshold = 0.6
predictions = (np.array(probabilities) >= threshold).astype(int)

# Evaluate the model
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)
roc_auc = roc_auc_score(y, probabilities)

print(f"Test Set Evaluation with Threshold = {threshold}:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")
