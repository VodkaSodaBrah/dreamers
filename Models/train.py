import os
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# Suppress warnings, set directory, etc.
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
os.chdir('/Users/mchildress/Code/dreamers/Analysis')

# Load and preprocess data
file_path = '/Users/mchildress/Code/dreamers/synthetic_time_series.csv'
data = pd.read_csv(file_path)
data = data.sort_values(by='x')

# Calculate the change from the previous step
data['y_diff'] = data['y'].diff()
data['y_binary'] = (data['y_diff'] > 0).astype(int)
data = data.dropna()

# Define the size of each set
train_size = int(len(data) * 0.6)
val_size = int(len(data) * 0.2)
test_size = len(data) - train_size - val_size

# Split data into training, validation, and test sets
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Extract features and labels for each set
X_train, y_train = train_data.drop(
    ['y', 'y_diff', 'y_binary'], axis=1), train_data['y_binary']
X_val, y_val = val_data.drop(
    ['y', 'y_diff', 'y_binary'], axis=1), val_data['y_binary']
X_test, y_test = test_data.drop(
    ['y', 'y_diff', 'y_binary'], axis=1), test_data['y_binary']

# Standardize features based on the training set only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

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
    n_iter=10,
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
    ['y', 'y_diff', 'y_binary'], axis=1).columns.tolist() + ['y_binary'])
test_data.to_csv('test_data.csv', index=False)

# Evaluate training performance
train_performance = random_search.best_score_
print("Best Cross-Validation Score from Randomized Search:", train_performance)

# Print the size of each set
print(f"Training set length: {len(X_train)}")
print(f"Validation set length: {len(X_val)}")
print(f"Test set length: {len(X_test)}")
