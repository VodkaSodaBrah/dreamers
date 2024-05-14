import os
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

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
joblib.dump(scaler, "scaler.pkl")

# Initialize the XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False,
                          eval_metric='logloss', reg_lambda=1, reg_alpha=0.5)

# Define the hyperparameter grid
param_distributions = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2]
}

# Cross-validation
tscv = TimeSeriesSplit(n_splits=5)
random_search = RandomizedSearchCV(
    model, param_distributions, n_iter=10, scoring='accuracy', n_jobs=-1, cv=tscv, verbose=1)
random_search.fit(X_scaled, y)

# Save the best model
best_model = random_search.best_estimator_
joblib.dump(best_model, "best_model.pkl")

print(
    f"Best Cross-Validation Score from Randomized Search: {random_search.best_score_}")
