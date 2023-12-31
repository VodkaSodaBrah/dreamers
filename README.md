# Time Series Analysis and Prediction with XGBoost

This repository contains code for analyzing and predicting time series data using the XGBoost machine learning framework. The project involves preprocessing time series data, creating lagged features, binary transformation, model training, and evaluation.

## Project Structure

- `time_series_analysis.py`: Main Python script with all the analysis and model training.
- `synthetic_time_series.csv`: Sample dataset used for the analysis.
- `/models`: Directory where trained models are saved.
- `/plots`: Directory for saved plots and data visualizations.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.11.5
- Pandas, NumPy, Matplotlib, XGBoost, scikit-learn, lime, shap

You can install these packages using `pip`:

```bash
pip3 install pandas numpy matplotlib xgboost scikit-learn lime shap

Installation

1. Clone the repository:
    git clone https://github.com/VodkaSodaBrah/dreamers.git
2. Navigate to the cloned directory:
    cd dreamers

Usage

Run the main script:
    python3 time_series_analysis.py

Features
Data Preprocessing: Includes standardization, lag features creation, and binary transformation based on median values.
Model Training: Uses XGBoost for training a classifier with hyperparameter tuning through grid search.
Model Evaluation: Evaluates model performance using accuracy, precision, recall, and F1 score.
Data Visualization: Generates plots for original data and prediction results.
Explainability: Implements SHAP and LIME for model interpretation.
Contributing
Contributions, issues, and feature requests are welcome. Feel free to check issues page if you want to contribute.

License
Distributed under the MIT License. See LICENSE for more information.

Contact
Michael Childress --- michael@michaelchildress.tech

Project Link: https://https://github.com/VodkaSodaBrah/dreamers

```
