import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

print("Starting script...")

# Ensure TensorFlow uses the GPU
physical_devices = tf.config.list_physical_devices('GPU')
print("Devices: ", physical_devices)

if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# Load and preprocess data
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

# Define the model creation function with regularization


def create_model(optimizer='adam', dropout_rate=0.5, l2_reg=0.01):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


print("Creating model...")

# Define parameter grid
param_grid = {
    'model__optimizer': ['adam', 'rmsprop'],
    'model__dropout_rate': [0.3, 0.5],
    'model__l2_reg': [0.001, 0.01]
}

# Wrap the Keras model with KerasClassifier for use in GridSearchCV
model = KerasClassifier(
    model=create_model, batch_size=16, epochs=50, verbose=1)

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    cv=StratifiedKFold(n_splits=3))
grid_result = grid.fit(X_train, y_train)

# Output the best hyperparameters
print(f"Best Hyperparameters: {grid_result.best_params_}")

# Evaluate the model on the validation set
val_pred_prob = grid_result.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred_prob)
print(f"Validation ROC-AUC: {val_auc}")

# Save the best model and scaler
grid_result.best_estimator_.model_.save('best_model.keras')
joblib.dump(scaler, "scaler.pkl")

print("Script completed.")
