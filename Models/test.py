import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

# Load the model and test data
best_model = joblib.load(
    "/Users/mchildress/Code/dreamers/Analysis/best_model.pkl")
test_data = pd.read_csv(
    '/Users/mchildress/Code/dreamers/Analysis/test_data.csv')

# Separate features and target
X_test = test_data.drop('y_binary', axis=1)
y_test = test_data['y_binary']

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
