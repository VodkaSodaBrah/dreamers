# /scripts/serve_model.py

import joblib
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request

app = Flask(__name__)


class NeuralNet(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5, l2_reg=0.01):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.l2_reg = l2_reg

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


model = NeuralNet(input_size=7)
model.load_state_dict(torch.load('/app/data/best_model.pth'))
model.eval()

scaler = joblib.load('/app/data/scaler.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not isinstance(data, list):
        return jsonify({'error': 'Invalid input format. Data should be a list of lists or a single list of features.'}), 400

    try:
        features = np.array([[d['lag1'], d['lag2'], d['lag3'], d['rolling_mean_3'],
                            d['rolling_std_3'], d['ema_3'], d['additional_feature']] for d in data])
    except KeyError as e:
        return jsonify({'error': f'Missing feature in input data: {e}'}), 400

    features_scaled = scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(features_tensor).squeeze().numpy()

    return jsonify(predictions.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
