import os
import optuna
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             precision_recall_fscore_support, roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import joblib  

print("Starting script...")

# ------------------------------------------------------------------------------
# 1) Global Settings
# ------------------------------------------------------------------------------
MAX_EPOCHS_PER_TRIAL = 15     
PATIENCE = 3                  
MAX_FINAL_EPOCHS = 50         
BATCH_SIZE = 16              

# 2) Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------------------
# 3) Load & Preprocess Data
# ------------------------------------------------------------------------------
file_path = '/Users/mchildress/Code/dreamers/data/synthetic_time_series.csv'
data = pd.read_csv(file_path)
data = data.sort_values(by='x').reset_index(drop=True)

# (A) Label for the NEXT step’s change
data['y_binary'] = (data['y'].shift(-1) > data['y']).astype(int)

# (B) Additional Features
data['y_diff'] = data['y'].diff()
data['rolling_mean_5_diff'] = data['y_diff'].rolling(5).mean()
data['rolling_ema_3_diff']  = data['y_diff'].ewm(span=3, adjust=False).mean()

# Example cyclical time-based features (simulated “day_of_week”)
data['day_of_week'] = data.index % 7
data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

# Pre-existing rolling stats and lag features
for lag in range(1, 6):
    data[f'lag{lag}'] = data['y'].shift(lag)

data['rolling_mean_3']  = data['y'].rolling(window=3).mean()
data['rolling_std_3']   = data['y'].rolling(window=3).std()
data['rolling_mean_5']  = data['y'].rolling(window=5).mean()
data['rolling_std_5']   = data['y'].rolling(window=5).std()
data['rolling_mean_10'] = data['y'].rolling(window=10).mean()
data['rolling_std_10']  = data['y'].rolling(window=10).std()
data['ema_3']           = data['y'].ewm(span=3, adjust=False).mean()

# Drop rows with NaN from rolling/lag or the last row from shift(-1)
data.dropna(inplace=True)

# Separate features vs. labels
# We drop 'x', 'y', 'y_binary', and 'day_of_week'
X = data.drop(['x', 'y', 'y_binary', 'day_of_week'], axis=1)
y = data['y_binary'].values

# ------------------------------------------------------------------------------
# 4) Sequential Splitting (~70% / 15% / 15%), no shuffle => time-ordered
# ------------------------------------------------------------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=False
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, shuffle=False
)

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

# ------------------------------------------------------------------------------
# 5) Scale Features & Convert to Float32 (for MPS)
# ------------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Convert to float32 for MPS
X_train_scaled = X_train_scaled.astype(np.float32)
X_val_scaled   = X_val_scaled.astype(np.float32)
X_test_scaled  = X_test_scaled.astype(np.float32)

# Convert labels to float32 if you're passing them as floats
y_train = y_train.astype(np.float32)
y_val   = y_val.astype(np.float32)
y_test  = y_test.astype(np.float32)

# ------------------------------------------------------------------------------
# 6) Sequence Dataset & DataLoader
# ------------------------------------------------------------------------------
class SequenceDataset(Dataset):
    """
    Creates sequences of length seq_len from row-wise data.
    Each sample is (seq_len, num_features) => label is the last row in that sequence.
    """
    def __init__(self, X_data, y_data, seq_len):
        self.X_data = X_data
        self.y_data = y_data
        self.seq_len = seq_len
        self.num_samples = len(X_data)

    def __len__(self):
        return self.num_samples - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X_data[idx : idx + self.seq_len, :]
        y_label = self.y_data[idx + self.seq_len - 1]
        return X_seq, y_label

def create_dataloader(X_array, y_array, seq_len, batch_size, shuffle=False):
    dataset = SequenceDataset(X_array, y_array, seq_len)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

# ------------------------------------------------------------------------------
# 7) LSTM Model Definition
# ------------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.0, bidirectional=False):
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
        # x shape: (batch_size, seq_len, input_dim)
        output, (h_n, c_n) = self.lstm(x)
        # Last hidden state from final time step
        last_hidden = output[:, -1, :]  # shape: (batch_size, out_dim)
        logits = self.fc(last_hidden)   # shape: (batch_size, 1)
        probs = self.sigmoid(logits).squeeze()  # shape: (batch_size,)
        return probs

# ------------------------------------------------------------------------------
# 8) Threshold Tuning Helper
# ------------------------------------------------------------------------------
def find_best_threshold(val_targets, val_scores):
    """
    Return threshold that maximizes F1 on the validation set.
    """
    best_thr = 0.5
    best_f1 = 0.0
    for thr in np.linspace(0.01, 0.99, 99):
        preds = (val_scores >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            val_targets, preds, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1

# ------------------------------------------------------------------------------
# 9) Optuna Objective
# ------------------------------------------------------------------------------
def objective(trial):
    """
    We define the search space and train for a limited # of epochs (MAX_EPOCHS_PER_TRIAL),
    using early stopping (PATIENCE). We'll track best Val AUC.
    """
    # (A) Search space
    seq_len = trial.suggest_int("seq_len", 5, 30, step=5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout    = trial.suggest_float("dropout", 0.0, 0.6, step=0.1)
    lr         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    bidir      = trial.suggest_categorical("bidirectional", [False, True])

    # (B) Model
    input_dim = X_train_scaled.shape[1]
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidir
    ).to(device)

    # (C) DataLoaders
    train_loader = create_dataloader(X_train_scaled, y_train, seq_len, BATCH_SIZE, shuffle=True)
    val_loader   = create_dataloader(X_val_scaled,   y_val,   seq_len, BATCH_SIZE, shuffle=False)

    # (D) Optimizer + Loss (with weight decay for reg)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_val_auc = 0.0
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS_PER_TRIAL):
        # --- Train ---
        model.train()
        total_loss = 0.0
        batches = 0
        for X_batch, y_batch in train_loader:
            # X_batch shape: (batch_size, seq_len, num_features)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        # --- Validation ---
        model.eval()
        val_preds = []
        val_targets_list = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                out = model(X_batch)
                val_preds.extend(out.cpu().numpy())
                val_targets_list.extend(y_batch.cpu().numpy())

        val_auc = roc_auc_score(val_targets_list, val_preds)

        # Early stopping within trial
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            # Save the best checkpoint for this trial
            torch.save(model.state_dict(), "best_model_checkpoint.pth")
            print(f"Checkpoint saved at epoch {epoch+1} with Val AUC: {val_auc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

        trial.report(val_auc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_auc

# ------------------------------------------------------------------------------
# 10) Run Optuna Study
# ------------------------------------------------------------------------------
import optuna

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

print("\nStudy complete!")
print(f"Best trial: {study.best_trial.number}")
print(f"Hyperparams: {study.best_trial.params}")
print(f"Best Val AUC: {study.best_value:.4f}")

joblib.dump(study, "optuna_study.pkl")
print("Optuna study saved to optuna_study.pkl")

# ------------------------------------------------------------------------------
# 11) Re-train Final Model with Best Hyperparams & Evaluate
# ------------------------------------------------------------------------------
best_params = study.best_trial.params
print("\n--- Re-training final model with best hyperparams ---")

seq_len   = best_params["seq_len"]
hidden_dim = best_params["hidden_dim"]
num_layers = best_params["num_layers"]
dropout    = best_params["dropout"]
lr         = best_params["lr"]
bidir      = best_params["bidirectional"]

input_dim = X_train_scaled.shape[1]

final_model = LSTMModel(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
    bidirectional=bidir
).to(device)

train_loader = create_dataloader(X_train_scaled, y_train, seq_len, BATCH_SIZE, shuffle=True)
val_loader   = create_dataloader(X_val_scaled,   y_val,   seq_len, BATCH_SIZE, shuffle=False)

criterion = nn.BCELoss()
optimizer = optim.Adam(final_model.parameters(), lr=lr, weight_decay=1e-4)

best_val_auc_final = 0.0
epochs_no_improve_final = 0
patience_final = 5  

for epoch in range(MAX_FINAL_EPOCHS):
    # --- Training ---
    final_model.train()
    total_loss = 0.0
    batches = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        preds = final_model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1

    # --- Validation ---
    final_model.eval()
    val_preds_epoch = []
    val_targets_epoch = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            out = final_model(X_batch)
            val_preds_epoch.extend(out.cpu().numpy())
            val_targets_epoch.extend(y_batch.cpu().numpy())

    val_auc = roc_auc_score(val_targets_epoch, val_preds_epoch)
    if val_auc > best_val_auc_final:
        best_val_auc_final = val_auc
        epochs_no_improve_final = 0
        torch.save(final_model.state_dict(), "best_model.pth")
        print(f"Final model saved at epoch {epoch+1} with Val AUC: {val_auc:.4f}")
    else:
        epochs_no_improve_final += 1
        if epochs_no_improve_final >= patience_final:
            print(f"Early stopping final training at epoch {epoch+1}")
            break

    avg_train_loss = total_loss / (batches if batches > 0 else 1)
    print(f"Epoch [{epoch+1}/{MAX_FINAL_EPOCHS}] "
          f"Train Loss={avg_train_loss:.4f}, Val AUC={val_auc:.4f} (best={best_val_auc_final:.4f})")

print(f"Final model best Val AUC after re-training = {best_val_auc_final:.4f}")

# ------------------------------------------------------------------------------
# 12) Threshold Tuning on Validation Set
# ------------------------------------------------------------------------------
val_preds_final = []
val_targets_final = []
final_model.eval()
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        out = final_model(X_batch)
        val_preds_final.extend(out.cpu().numpy())
        val_targets_final.extend(y_batch.cpu().numpy())

val_preds_final = np.array(val_preds_final)
val_targets_final = np.array(val_targets_final)
best_threshold, best_f1_val = find_best_threshold(val_targets_final, val_preds_final)
print(f"\nBest threshold on Val for max F1: {best_threshold:.3f}, F1={best_f1_val:.3f}")

# ------------------------------------------------------------------------------
# 13) Evaluate on Test
# ------------------------------------------------------------------------------
test_dataset = SequenceDataset(X_test_scaled, y_test, seq_len)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_preds = []
test_targets_list = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        out = final_model(X_batch)
        test_preds.extend(out.cpu().numpy())
        test_targets_list.extend(y_batch.numpy())

test_preds = np.array(test_preds)
test_targets_list = np.array(test_targets_list)

test_auc = roc_auc_score(test_targets_list, test_preds)
test_pred_classes = (test_preds >= best_threshold).astype(int)

cm = confusion_matrix(test_targets_list, test_pred_classes)
tn, fp, fn, tp = cm.ravel()
precision, recall, f1, _ = precision_recall_fscore_support(
    test_targets_list, test_pred_classes, average='binary', zero_division=0
)

print("\n--- Final Test Evaluation ---")
print(f"Test AUC: {test_auc:.4f}")
print("Confusion Matrix:", cm)
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f} (threshold={best_threshold:.3f})")

fpr, tpr, _ = roc_curve(test_targets_list, test_preds)
test_roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {test_roc_auc:.3f})")
plt.plot([0, 1], [0, 1], 'r--')
plt.title("Test ROC Curve (LSTM + Optuna + Extended Features)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("test_roc_optuna.png")
plt.close()

# ------------------------------------------------------------------------------
# 14) Save scaler + final test set
# ------------------------------------------------------------------------------
joblib.dump(scaler, 'scaler_optuna.pkl')
print("Scaler saved as scaler_optuna.pkl")

X_test_copy = X_test.copy()
X_test_copy['y_binary'] = y_test
X_test_copy.to_csv("test_only_optuna.csv", index=False)
print("Saved test set as test_only_optuna.csv")

print("Script completed with a full pipeline, large Optuna search, early stopping, threshold tuning, float32 conversions for MPS, and model checkpointing/saving.")