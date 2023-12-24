import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, PolynomialFeatures,
                                   StandardScaler)

# Load the data from a CSV file
dreamers_sample = pd.read_csv(
    '/Users/mchildress/Code/dreamers/synthetic_time_series.csv')

# Assuming 'y' is your target variable and 'x' are your features
X = dreamers_sample.drop('y', axis=1)
y = dreamers_sample['y']

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Build preprocessor for numeric and categorical data
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Test different degrees of polynomials and use cross-validation to find the best one
best_degree = 1
best_score = -np.inf
for degree in range(1, 6):
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('polynomial', PolynomialFeatures(degree=degree)),
        ('regressor', LinearRegression())
    ])
    scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=5)
    score = scores.mean()
    print(f"Degree: {degree}, Cross-Validated R-squared: {score:.3f}")

    if score > best_score:
        best_score = score
        best_degree = degree

# Retrain the model using the best degree found
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('polynomial', PolynomialFeatures(degree=best_degree)),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)
test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(f"Test MSE with best degree ({best_degree}): {test_mse:.3f}")
print(f"Test R-squared with best degree ({best_degree}): {test_r2:.3f}")

# Plotting the regression curve using the best_degree
x_min = X_train[numeric_features].min().min()
x_max = X_train[numeric_features].max().max()
x_range = np.linspace(x_min, x_max, 100)
feature_name = numeric_features[0]  # Replace with your actual feature name
X_range = pd.DataFrame(x_range, columns=[feature_name])
y_range = model.predict(X_range)

plt.scatter(X_train[feature_name], y_train,
            color='lightblue', label='Training data')
plt.scatter(X_test[feature_name], y_test, color='green', label='Test data')
plt.plot(X_range[feature_name], y_range, color='red',
         label='Polynomial regression line')
plt.xlabel('Scaled Feature Values')
plt.ylabel('Target Value')
plt.title('Polynomial Regression Fit')
plt.legend()
plt.show()
