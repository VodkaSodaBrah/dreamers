import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import StandardScaler

# Load the model and test data
best_model = joblib.load(
    "/Users/mchildress/Code/dreamers/Analysis/best_model.pkl")
test_data = pd.read_csv(
    "/Users/mchildress/Code/dreamers/Analysis/test_data.csv")

# Ensure that 'y' is present
if 'y' not in test_data.columns:
    raise ValueError(
        "Column 'y' is missing from the test data. Please check the data preparation step in train.py.")

# Replicate feature engineering from train.py
test_data['y_diff'] = test_data['y'].diff()
test_data['lag1'] = test_data['y'].shift(1)
test_data['lag2'] = test_data['y'].shift(2)
test_data['lag3'] = test_data['y'].shift(3)
test_data['rolling_mean_3'] = test_data['y'].rolling(window=3).mean()
test_data['rolling_std_3'] = test_data['y'].rolling(window=3).std()
test_data['ema_3'] = test_data['y'].ewm(span=3, adjust=False).mean()

# Drop rows with NA values that may have been introduced by shifts or rolling functions
test_data.dropna(inplace=True)

# Extract features and target
X_test = test_data.drop(['y', 'y_diff', 'y_binary'], axis=1)
y_test = test_data['y_binary']

# Load the scaler from the training phase to ensure consistent scaling
scaler = joblib.load("/Users/mchildress/Code/dreamers/Analysis/scaler.pkl")
X_test = scaler.transform(X_test)

# Predict and evaluate
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print('Test Set Evaluation:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print('Confusion Matrix:')
print(cm)
