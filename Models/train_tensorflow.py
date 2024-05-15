import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

print("Starting script...")

# Ensure TensorFlow uses the GPU
print("Devices: ", tf.config.list_physical_devices('GPU'))

# Load and preprocess data
# Ensure this path is correct
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

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Check the distribution of the classes after resampling
print(f"Class distribution after resampling: {Counter(y_resampled)}")

# Split data into training and validation sets
print("Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define the model creation function


def create_model(optimizer='adam', dropout_rate=0.5):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


print("Creating model...")

# Define hyperparameters to manually search
batch_sizes = [16, 32, 64]
epochs = [50, 100]
optimizers = ['adam', 'rmsprop']
dropout_rates = [0.3, 0.5, 0.7]

best_auc = 0
best_params = {}

print("Starting manual grid search...")

# Manual grid search
for batch_size in batch_sizes:
    for epoch in epochs:
        for optimizer in optimizers:
            for dropout_rate in dropout_rates:

                # Create and compile the model
                model = create_model(optimizer=optimizer,
                                     dropout_rate=dropout_rate)

                # Define early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True)

                # Train the model
                history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(
                    X_val, y_val), callbacks=[early_stopping])

                # Evaluate the model on the validation set
                val_pred_prob = model.predict(X_val, batch_size=batch_size)
                val_auc = roc_auc_score(y_val, val_pred_prob)

                print(f"Validation ROC-AUC: {val_auc}")

                # Update best model and parameters if current model is better
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_params = {
                        'batch_size': batch_size,
                        'epochs': epoch,
                        'optimizer': optimizer,
                        'dropout_rate': dropout_rate
                    }
                    best_model = model

print(f"Best Validation ROC-AUC: {best_auc}")
print(f"Best Hyperparameters: {best_params}")

# Save the best model and scaler
best_model.save('best_model.keras')  # Save in the recommended .keras format
joblib.dump(scaler, "scaler.pkl")

print("Script completed.")
