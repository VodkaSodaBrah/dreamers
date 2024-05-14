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
model_path = "best_model.pkl"
scaler_path = "scaler.pkl"
best_model, scaler = load_model_and_scaler(model_path, scaler_path)

# Load the test data
test_data_path = '/Users/mchildress/Code/dreamers/Analysis/hold_out_data.csv'
test_data = pd.read_csv(test_data_path)

# Preprocess the test data
X_test = test_data.drop(["y_binary"], axis=1)
y_test = test_data["y_binary"]
X_test_scaled = scaler.transform(X_test)

# Evaluate the model
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

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
print(classification_report(y_test, y_pred))
