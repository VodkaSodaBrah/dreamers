import os
import warnings

import joblib  # For saving the model
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# Suppress warnings, set directory, etc.
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
os.chdir('/Users/mchildress/Code/dreamers/Analysis')

# Load and preprocess data
file_path = '/Users/mchildress/Code/dreamers/synthetic_time_series.csv'
data = pd.read_csv(file_path)
data = data.sort_values(by='x')
threshold = data['y'].median()
data['y_binary'] = (data['y'] > threshold).astype(int)
for lag in [1, 2, 3]:
    data[f'lag_{lag}'] = data['y'].shift(lag)
window_size = 3
data['rolling_mean'] = data['y'].rolling(window=window_size).mean()
data['rolling_std'] = data['y'].rolling(window=window_size).std()
period = 12
for k in range(1, 4):
    data[f'sin_{k}'] = np.sin(2 * np.pi * k * data['x'] / period)
    data[f'cos_{k}'] = np.cos(2 * np.pi * k * data['x'] / period)
data.dropna(inplace=True)

# Define features and target
X = data.drop(['y', 'y_binary'], axis=1)
y_binary = data['y_binary']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_binary, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define the hyperparameter grid
param_distributions = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.5, 0.8, 1]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=10,  # Number of parameter settings that are sampled
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42)

# Fit the model with early stopping
random_search.fit(X_train, y_train,
                  early_stopping_rounds=10,
                  eval_set=[(X_val, y_val)],
                  verbose=True)

# Save the best model and test data
best_model = random_search.best_estimator_
joblib.dump(best_model, "best_model.pkl")
test_data = pd.DataFrame(np.column_stack((X_test, y_test)), columns=data.drop(
    ['y', 'y_binary'], axis=1).columns.tolist() + ['y_binary'])
test_data.to_csv('test_data.csv', index=False)

# Evaluate training performance
train_performance = random_search.best_score_
print("Best Cross-Validation Score from Randomized Search:", train_performance)
