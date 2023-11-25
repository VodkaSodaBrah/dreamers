# Importing necessary libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the data from a CSV file
dreamers_sample = pd.read_csv(
    '/Users/mchildress/Code/dreamers/synthetic_time_series.csv')

# Assuming 'y' is your target variable and 'x' is your feature.
X = dreamers_sample.drop('y', axis=1)
y = dreamers_sample['y']

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

# Convert linear regression predictions to binary for classification
binary_predictions = [1 if val >= 0.5 else 0 for val in y_pred]
y_test_binary = [1 if val >= 0.5 else 0 for val in y_test]

# Calculate accuracy
accuracy = accuracy_score(y_test_binary, binary_predictions)

# Print the results
if accuracy > 0.5:
    print(
        f"Model's accuracy of {accuracy*100:.2f}% is better than random guessing.")
elif accuracy == 0.5:
    print("Model's accuracy is equivalent to random guessing.")
else:
    print(
        f"Model's accuracy of {accuracy*100:.2f}% is worse than random guessing.")

# Calculate mean squared error
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')

# For simplicity, let's assume X has only one feature for plotting.
# Extract the values of that feature for the x-axis
x_values = X[numeric_features[0]].values if len(numeric_features) > 0 else None

# Generate plot using Plotly for the raw data
fig = go.Figure()

# Add scatter plot for raw data
if x_values is not None:
    fig.add_trace(go.Scatter(x=x_values, y=y, mode='markers', name='Raw Data'))

# Set labels and title
fig.update_layout(title='Raw Data Visualization',
                  xaxis_title='Feature Values',
                  yaxis_title='Target Values')

# Show the plot
fig.show()
