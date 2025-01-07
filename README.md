# Time Series Analysis and Prediction with Machine Learning

This repository contains code for analyzing and predicting time series data using machine learning models, specifically focusing on time-series forecasting. The project involves sequential data preprocessing, feature engineering, binary classification, model training with hyperparameter optimization, evaluation, and result visualization.

## Project Structure

- `train_pytorch.py`: Script for training models using PyTorch with hyperparameter optimization via Optuna.
- `test_pytorch.py`: Script for testing PyTorch-trained models and evaluating performance.
- `synthetic_time_series.csv`: Sample dataset used for analysis.
- `/Analysis`: Directory where trained models are saved.
- `/Logs`: Directory where training and testing logs are saved.
- `/Plots`: Directory where visualizations (e.g., ROC curves) are stored.

## Key Features

### Data Preprocessing
- Sequential splitting of data to maintain time-series integrity and avoid lookahead bias.
- Standardization of features using `StandardScaler`.
- Feature engineering includes:
  - Lagged features.
  - Rolling statistics (mean, standard deviation).
  - Exponential moving averages (EMA).
  - Cyclical time-based features (e.g., day of the week).

### Model Training
- Utilizes an LSTM (Long Short-Term Memory) architecture with customizable parameters:
  - Sequence length, hidden dimensions, number of layers, dropout rate, and bidirectionality.
- Hyperparameter optimization using `Optuna`.
- Early stopping to prevent overfitting during training.

### Model Evaluation
- Evaluates performance using metrics:
  - Area Under the Curve (AUC), precision, recall, F1 score.
- Confusion matrix for detailed classification analysis.
- Threshold tuning on the validation set for optimal F1 score.
- Visualizes ROC curves for performance insights.

### Logging and Checkpoints
- Saves training checkpoints for best validation AUC during each trial.
- Final model saved after retraining with optimal hyperparameters.
- Logs results to enable reproducibility and easy comparison.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.10 or 3.11
- Required libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`, `imbalanced-learn`, `torch`, `optuna`.

Install the necessary dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn joblib imbalanced-learn torch optuna

Installation

Clone the repository:

git clone https://github.com/VodkaSodaBrah/dreamers.git

Navigate to the directory:

cd dreamers

Usage

Training the Model

Run the PyTorch training script:

python scripts/train_pytorch.py

This script:
	•	Preprocesses the data.
	•	Creates sequences for LSTM input.
	•	Performs hyperparameter optimization using Optuna.
	•	Retrains the final model using the best hyperparameters.

Testing the Model

Run the testing script:

python scripts/test_pytorch.py

This script:
	•	Loads the saved model and scaler.
	•	Evaluates the model on the test set.
	•	Outputs metrics including AUC, precision, recall, F1 score, and saves a ROC curve plot.

Contributing

Contributions, issues, and feature requests are welcome. Please open an issue or pull request for discussion.

License

Distributed under the MIT License. See LICENSE for more information.

Contact

Michael Childress - mchildress@me.com

Project Link: https://github.com/VodkaSodaBrah/dreamers