# Time Series Analysis and Prediction with Machine Learning

This repository contains code for analyzing and predicting time series data using machine learning models. The project involves preprocessing time series data, creating lagged features, binary transformation, model training, evaluation, and logging results.

## Project Structure

- `train.py`: Script for training models using classical machine learning algorithms.
- `test.py`: Script for testing models trained using `train.py`.
- `train_tensorflow.py`: Script for training models using TensorFlow on GPU.
- `test_tensorflow.py`: Script for testing models trained using `train_tensorflow.py`.
- `log_results.py`: Script to run training and testing scripts and log the results.
- `synthetic_time_series.csv`: Sample dataset used for the analysis.
- `/Analysis`: Directory where trained models are saved.
- `/Logs`: Directory where log files with training and testing results are saved.

## Getting Started

### Prerequisites

```
Ensure you have the following installed:

- Python 3.10.14 or 3.11
- Pandas, NumPy, Matplotlib, scikit-learn, joblib, imbalanced-learn, TensorFlow, and other required libraries.

You can install these packages using `pip`:

pip3 install pandas numpy matplotlib scikit-learn joblib imbalanced-learn tensorflow
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/VodkaSodaBrah/dreamers.git
Navigate to the cloned directory:
bash
Copy code
cd dreamers

Usage

Training and Testing with Classical Machine Learning

Run the training script:

python3 train.py
Ensure the path to the sample dataset is set correctly.

Run the testing script:

python3 test.py
Training and Testing with TensorFlow
Run the TensorFlow training script:

python3 train_tensorflow.py
Ensure the path to the sample dataset is set correctly.

Run the TensorFlow testing script:

python3 test_tensorflow.py
Logging Results
To run the training and testing scripts and log the results:

python3 log_results.py
This will run train.py, test.py, train_tensorflow.py, and test_tensorflow.py and log the outputs in the /Logs directory, organized by date and time.

Features

Data Preprocessing: Includes standardization, lag features creation, and binary transformation based on median values.
Model Training: Uses classical machine learning algorithms and TensorFlow for training classifiers with hyperparameter tuning.
Model Evaluation: Evaluates model performance using accuracy, precision, recall, F1 score, and ROC-AUC.
Data Visualization: Generates plots for original data and prediction results.
Logging: Logs training and testing results for tracking progress over time.
Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.

License
Distributed under the MIT License. See LICENSE for more information.

Contact
Michael Childress - mchildress@me.com

Project Link: https://github.com/VodkaSodaBrah/dreamers
```
