import os

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)


# Function to load model and scaler
def load_model_and_scaler(model_path, scaler_path):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        raise FileNotFoundError("Model or Scaler file not found.")


# Load the model and scaler
model_path = "/Users/mchildress/Code/dreamers/Models/best_model.pkl"
scaler_path = "/Users/mchildress/Code/dreamers/Models/scaler.pkl"
best_model, scaler = load_model_and_scaler(model_path, scaler_path)

# Load the test data
# Ensure this path is correct
test_data_path = '/Users/mchildress/Code/dreamers/Analysis/hold_out_data.csv'
test_data = pd.read_csv(test_data_path)

# Preprocess the test data
X_test = test_data.drop(["y_binary"], axis=1)
y_test = test_data["y_binary"]
X_test_scaled = scaler.transform(X_test)

# Evaluate the model
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

# Print evaluation metrics
evaluation_metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 score": f1
}
print("Test Set Evaluation:")
for metric_name, value in evaluation_metrics.items():
    print(f"{metric_name.capitalize()}: {value:.2f}")

# Print classification report
print(classification_report(y_test, y_pred, zero_division=1))

# Evaluate the model with different decision thresholds
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]  # Example thresholds
for threshold in thresholds:
    y_pred_adjusted = (y_prob >= threshold).astype(int)

    accuracy_adjusted = accuracy_score(y_test, y_pred_adjusted)
    precision_adjusted = precision_score(
        y_test, y_pred_adjusted, zero_division=1)
    recall_adjusted = recall_score(y_test, y_pred_adjusted, zero_division=1)
    f1_adjusted = f1_score(y_test, y_pred_adjusted, zero_division=1)

    # Print evaluation metrics with adjusted threshold
    print(f"Test Set Evaluation with Threshold = {threshold}:")
    evaluation_metrics_adjusted = {
        "Accuracy": accuracy_adjusted,
        "Precision": precision_adjusted,
        "Recall": recall_adjusted,
        "F1 score": f1_adjusted
    }
    for metric_name, value in evaluation_metrics_adjusted.items():
        print(f"{metric_name.capitalize()}: {value:.2f}")

    # Print classification report with adjusted threshold
    print(classification_report(y_test, y_pred_adjusted, zero_division=1))
