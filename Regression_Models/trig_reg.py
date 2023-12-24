import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Load the data from a CSV file
dreamers_sample = pd.read_csv(
    '/Users/mchildress/Code/dreamers/synthetic_time_series.csv')

# Define the sine function to fit


def sine_function(x, amplitude, frequency, phase_shift, vertical_shift):
    return amplitude * np.sin(frequency * x + phase_shift) + vertical_shift


# Extract x and y values for curve fitting
x_values = dreamers_sample['x'].values
y_values = dreamers_sample['y'].values

# Normalize x values to improve optimization performance
x_values_normalized = x_values / np.max(x_values)

# Use non-linear least squares to fit the sine function to the data
params, params_covariance = curve_fit(
    sine_function, x_values_normalized, y_values, p0=[1, 2 * np.pi, 0, 0])

# Generate predicted y-values using the optimized parameters
y_pred = sine_function(x_values_normalized, *params)

# Calculate R-squared for the fitted sine function
r2 = r2_score(y_values, y_pred)
print(f'R-squared: {r2:.2f}')

# Plot the raw data and the fitted sine curve using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dreamers_sample['x'], y=dreamers_sample['y'], mode='markers', name='Actual'))
fig.add_trace(go.Scatter(
    x=dreamers_sample['x'], y=y_pred, mode='lines', name='Fitted Sine Function'))
fig.update_layout(title='Fitting Sine Function to Data',
                  xaxis_title='x',
                  yaxis_title='y')
fig.show()
