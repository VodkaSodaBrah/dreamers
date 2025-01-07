import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for plotting
import matplotlib.pyplot as plt

import joblib
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, confusion_matrix, 
                             precision_recall_fscore_support, roc_curve, auc)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=2, dropout=0.0, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = output[:, -1, :]  
        logits = self.fc(last_hidden)
        probs = self.sigmoid(logits).squeeze()
        return probs

# 1) Load the scaler
scaler = joblib.load("scaler_optuna.pkl")
print("Scaler loaded successfully.")

# 2) Load the test CSV
test_data = pd.read_csv("test_only_optuna.csv")
print("Test set loaded successfully.")

# 3) Split features/labels
y_test = test_data["y_binary"]
X_test = test_data.drop("y_binary", axis=1)

if "Unnamed: 0" in X_test.columns:
    X_test = X_test.drop(columns=["Unnamed: 0"])

# 4) Scale the test data
X_test_scaled = scaler.transform(X_test)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)

# 5) Load the model with matching hyperparameters
input_dim = X_test.shape[1]  
hidden_dim = 512  
num_layers = 2    
dropout = 0.0     
bidirectional = True  

model = LSTMModel(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
    bidirectional=bidirectional
).to("cpu")  

# Load the trained model weights
state_dict = torch.load("best_model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully.")

# 6) Predictions
with torch.no_grad():
    # Ensure the input tensor has the correct shape [batch_size, seq_len, input_dim]
    if X_test_scaled.ndim == 2:  # Add sequence length dimension
        X_test_scaled = X_test_scaled.unsqueeze(1)  # [batch_size, 1, input_dim]
    predictions = model(X_test_scaled).numpy()

# 7) Evaluate the model
test_auc = roc_auc_score(y_test, predictions)
test_pred_classes = (predictions >= 0.5).astype(int)
cm = confusion_matrix(y_test, test_pred_classes)
tn, fp, fn, tp = cm.ravel()
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, test_pred_classes, average="binary"
)

print("\n--- Test Evaluation ---")
print(f"Test AUC: {test_auc:.4f}")
print("Confusion Matrix (TN, FP, FN, TP):", cm)
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

# 8) ROC Plot
fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc_val = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
plt.plot([0, 1], [0, 1], "r--")
plt.title("Test ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("test_roc_test_pytorch.png")
plt.close()
print("ROC curve saved as test_roc_test_pytorch.png.")