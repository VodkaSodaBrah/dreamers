import os

import lime
import lime.lime_tabular
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

os.chdir('/Users/mchildress/Code/dreamers')
matplotlib.use('Agg')

# Load the data
file_path = '/Users/mchildress/Code/dreamers/synthetic_time_series.csv'
data = pd.read_csv(file_path)

# Assuming 'x' is a time-based feature (like a timestamp), convert it if necessary
# For this example, I'm treating 'x' as an ordinal time feature

# Sort data by 'x' if it's time-related
data = data.sort_values(by='x')

# Convert the continuous target variable into binary
threshold = data['y'].median()
data['y_binary'] = (data['y'] > threshold).astype(int)

# Lagged terms
for lag in [1, 2, 3]:  # You can choose different lags
    data[f'lag_{lag}'] = data['y'].shift(lag)

# Rolling window statistics
window_size = 3  # You can choose different window sizes
data['rolling_mean'] = data['y'].rolling(window=window_size).mean()
data['rolling_std'] = data['y'].rolling(window=window_size).std()

# Fourier transformation to capture cyclical behavior
# You will need to determine the period based on your specific data
period = 12  # Placeholder for the actual period
for k in range(1, 4):  # First few Fourier terms
    data[f'sin_{k}'] = np.sin(2 * np.pi * k * data['x'] / period)
    data[f'cos_{k}'] = np.cos(2 * np.pi * k * data['x'] / period)

# Drop rows with missing values created by lags and rolling stats
data.dropna(inplace=True)

# Define features and target
X = data.drop(['y', 'y_binary'], axis=1)
y_binary = data['y_binary']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42)

# Fit the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define the hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1]
}

# Grid search
grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Use the best estimator to make predictions
best_model = grid_search.best_estimator_

# Save the model in JSON format
best_model = grid_search.best_estimator_
best_model.save_model("best_model.json")

# Make predictions
y_pred = best_model.predict(X_test)

# Calculate and print accuracy, precision, recall, and F1 score for the test set
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print('Test Set Evaluation:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print('Confusion Matrix:')
print(cm)

# Plotting the actual vs predicted results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test,
            color='blue', label='Actual', alpha=0.5)
plt.scatter(range(len(y_pred)), y_pred, color='red',
            label='Predicted', alpha=0.5)
plt.title('Actual vs Predicted Results')
plt.xlabel('Samples')
plt.ylabel('Binary Classification Results')
plt.legend()
plt.show()

# Initialize SHAP explainer
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train)

# SHAP summary plot
shap.summary_plot(shap_values, X_train, feature_names=data.columns, show=False)
plt.savefig("shap_summary_plot.png")
plt.close()


# Initialize LIME explainer with actual feature names
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    # List of actual feature names
    feature_names=data.drop(['y', 'y_binary'], axis=1).columns.tolist(),
    class_names=['class_0', 'class_1'],  # Adjust based on your target variable
    mode='classification'
)

# Explain the prediction for a specific instance
instance_index = 1  # Change as needed
lime_exp = explainer.explain_instance(
    X_test[instance_index],
    best_model.predict_proba,
    num_features=len(X.columns)
)

# Visualize the explanation
fig = lime_exp.as_pyplot_figure()
fig.savefig("lime_explanation.png")
plt.close()
