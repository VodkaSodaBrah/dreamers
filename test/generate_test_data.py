# /test/generate_test_data.py

import json
import random

test_data = []
for _ in range(1000):
    entry = {
        "lag1": random.uniform(0, 1),
        "lag2": random.uniform(0, 1),
        "lag3": random.uniform(0, 1),
        "rolling_mean_3": random.uniform(0, 1),
        "rolling_std_3": random.uniform(0, 1),
        "ema_3": random.uniform(0, 1),
        "additional_feature": random.uniform(0, 1)
    }
    test_data.append(entry)

with open('/app/Test/test_data_1000.json', 'w') as f:
    json.dump(test_data, f)

print("Test data generated and saved to /app/test/test_data_1000.json")
