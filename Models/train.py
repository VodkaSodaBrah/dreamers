import os
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Suppress warnings
warnings.filterwarnings('ignore')

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
joblib.dump(scaler, "scaler.pkl")

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Check the distribution of the classes after resampling
print(f"Class distribution after resampling: {Counter(y_resampled)}")

# Initialize classifiers
log_clf = LogisticRegression(max_iter=5000)
svc_clf = SVC(probability=True)
rf_clf = RandomForestClassifier()

# Create a voting classifier
voting_clf = VotingClassifier(estimators=[
    ('lr', log_clf),
    ('svc', svc_clf),
    ('rf', rf_clf)
], voting='soft')

# Define the hyperparameter grid for the voting classifier
param_distributions = {
    'lr__C': [0.01, 0.1, 1, 10, 100],
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'svc__gamma': ['scale', 'auto'],
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

# Define custom scoring function


def custom_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='binary')


scorer = make_scorer(custom_f1_score, greater_is_better=True)

# Cross-validation
skf = StratifiedKFold(n_splits=5)
random_search = RandomizedSearchCV(
    voting_clf, param_distributions, n_iter=20, scoring=scorer, n_jobs=-1, cv=skf, verbose=1)
random_search.fit(X_resampled, y_resampled)

# Save the best model
best_model = random_search.best_estimator_
joblib.dump(best_model, "best_model.pkl")

print(
    f"Best Cross-Validation Score from Randomized Search: {random_search.best_score_}")
