# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the data from a CSV file
dreamers_sample = pd.read_csv(
    '/Users/mchildress/Code/dreamers/synthetic_time_series.csv')

# Assuming 'y' is your target variable and 'x' is your feature.
X = dreamers_sample.drop('y', axis=1)
y = dreamers_sample['y']

# Calculate the MSE for predicting the center (mean) of the y values for all instances
center_value = 1010  # approximate center of y values
mse_center = ((dreamers_sample['y'] - center_value) ** 2).mean()
print(f'MSE for predicting center value: {mse_center}')

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Build preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline to make it a regressor
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', LinearRegression())])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Fit the model on training data and make predictions on test data
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calculate mean squared error
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')

# Cross-validation
scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-validation scores: {-scores}')
print(f'Average cross-validation score: {-np.mean(scores)}')

# Generate plot
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Linear Regression Model: Prediction vs Actual')
plt.grid(True)

# Plot ideal prediction line
plt.plot([y_test.min(), y_test.max()], [
         y_test.min(), y_test.max()], lw=2, color='red')

# Adjust the scale
plt.xlim([600, 1325])
plt.ylim([1000, 1018])

plt.show()
