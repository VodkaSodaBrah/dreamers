import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
file_path = 'synthetic_time_series.csv'
data = pd.read_csv(file_path)
data = data.sort_values(by='x')

# Calculate the change from the previous step
data['y_diff'] = data['y'].diff()
data['y_binary'] = (data['y_diff'] > 0).astype(int)
data = data.dropna()

# Feature engineering
data['lag1'] = data['y'].shift(1)
data['lag2'] = data['y'].shift(2)
data['lag3'] = data['y'].shift(3)
data['rolling_mean_3'] = data['y'].rolling(window=3).mean()
data['rolling_std_3'] = data['y'].rolling(window=3).std()
data['ema_3'] = data['y'].ewm(span=3, adjust=False).mean()
data = data.dropna()

# Extract features and target
X = data.drop(['y', 'y_diff', 'y_binary'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved successfully.")
