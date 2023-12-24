import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the data from a CSV file
# Replace this with the path to your actual CSV file
file_path = '/Users/mchildress/Code/dreamers/synthetic_time_series.csv'
data = pd.read_csv(file_path)

# Assuming 'y' is your target variable and 'x' is your feature
X = data[['x']]  # Assuming 'x' is the name of your feature column
y = data['y']

# Since exponential regression is a linear regression on the log-transformed target,
# we'll apply a natural log transformation to 'y'
y_transformed = np.log(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_transformed, test_size=0.2, random_state=42)

# Fit the linear regression model on the transformed target
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set and transform predictions back
y_pred_transformed = model.predict(X_test)
y_pred = np.exp(y_pred_transformed)

# Calculate mean squared error and R-squared for the predictions on the original scale
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')
