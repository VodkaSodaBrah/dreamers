import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LinearGAM, f, l, s
from pygam.terms import TermList
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the data from a CSV file
dreamers_sample = pd.read_csv(
    '/Users/mchildress/Code/dreamers/synthetic_time_series.csv')

# Assuming 'y' is your target variable and the rest are your features
X = dreamers_sample.drop('y', axis=1)
y = dreamers_sample['y']

# Save the column names for later use in plotting
feature_names = X.columns.tolist()

# Identify numerical and categorical columns if present
# This is important as GAMs require specifying the type of each feature
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Convert categorical columns to category type if not already
for col in categorical_features:
    X[col] = X[col].astype('category')

# We need to create a TermList for our GAM model that specifies the type of each feature
# 's' for spline/numeric features and 'f' for factor/categorical features
# The TermList is constructed by starting with an intercept using 'l(0)' and adding other terms
terms = l(0)
for i, col in enumerate(feature_names):
    if col in numeric_features:
        terms += s(i)
    elif col in categorical_features:
        terms += f(i)

# Convert X to a numpy array for pygam
X = X.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Fit a GAM with the specified terms
# Use gridsearch to find the optimal lambda for smoothing
gam = LinearGAM(terms=terms).gridsearch(X_train, y_train)

# Predict on the test set
y_pred = gam.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R-squared: {r2:.3f}')
print(f'Mean Squared Error: {mse:.3f}')

# Plot partial dependence for each feature
for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue

    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, width=0.95)

    plt.figure()
    plt.plot(XX[:, i], pdep)
    plt.plot(XX[:, i], confi, c='r', ls='--')
    # Use the saved feature names for labeling
    plt.title(f'Partial dependence for feature {feature_names[i]}')
    plt.xlabel(feature_names[i])
    plt.ylabel('Response')
    plt.show()
