import os
import warnings

import joblib  # For saving the model
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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
data = data.dropna()  # Remove the first row since it will have a NaN value for diff

# Feature engineering
# Lag features
data['lag1'] = data['y'].shift(1)
data['lag2'] = data['y'].shift(2)
data['lag3'] = data['y'].shift(3)

# Rolling window statistics
data['rolling_mean_3'] = data['y'].rolling(window=3).mean()
data['rolling_std_3'] = data['y'].rolling(window=3).std()

# Exponential Moving Average
data['ema_3'] = data['y'].ewm(span=3, adjust=False).mean()

# Ensure no NA values introduced by new features
data = data.dropna()

# Extract features and target
X = data.drop(['y', 'y_diff', 'y_binary'], axis=1)
y = data['y_binary']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use in testing
joblib.dump(scaler, "/Users/mchildress/Code/dreamers/Analysis/scaler.pkl")

# Initialize the XGBoost classifier with refined parameters
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    reg_lambda=1,  # L2 regularization to reduce overfitting
    reg_alpha=0.5,  # L1 regularization to further penalize large coefficients
)

# Define the hyperparameter grid with reduced complexity
param_distributions = {
    'max_depth': [3, 5, 7],  # Reduced max depth to limit tree size
    # Lower learning rates for more gradual learning
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 150],  # Fewer estimators to prevent overfitting
    'subsample': [0.5, 0.7, 0.9],  # Subsampling to introduce stochasticity
}

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Initialize RandomizedSearchCV with TimeSeriesSplit
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=10,  # Number of parameter settings that are sampled
    scoring='accuracy',
    cv=tscv,
    verbose=1,
    random_state=42)

# Fit the model
random_search.fit(X_scaled, y)

# Save the best model
best_model = random_search.best_estimator_
joblib.dump(best_model, "/Users/mchildress/Code/dreamers/Analysis/best_model.pkl")

# Evaluate training performance
train_performance = random_search.best_score_
print("Best Cross-Validation Score from Randomized Search:", train_performance)
