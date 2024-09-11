# XGBoost Regressor with RandomizedSearchCV (GPU-Accelerated) & Autogluon Regressor (GPU-Accelerated)
## This model was trained on an NVIDIA A100 GPU
**Special thanks to Aldin Risa for his contribution**
https://www.linkedin.com/in/drapraks/
https://www.linkedin.com/in/aldin-risa-304655299/

This repository contains an implementation of an XGBoost regression model trained using RandomizedSearchCV for hyperparameter tuning. The model utilizes GPU acceleration for faster training and cross-validation. The dataset is related to car prices, and the task is to predict the price of a car based on its features.

## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)

## Installation

To run this notebook, you need to have the following libraries installed:

```bash
pip install xgboost scikit-learn pandas numpy
```

### GPU Support

Ensure that your environment has access to a CUDA-compatible GPU and that the necessary drivers and libraries are installed for GPU acceleration with XGBoost.

- **XGBoost** requires CUDA for GPU support.
- Set up the GPU environment (e.g., NVIDIA drivers, CUDA toolkit) before running this notebook.

## Features

- **XGBoost Regressor**: A powerful gradient-boosting model that uses GPU acceleration for fast training.
- **RandomizedSearchCV**: Used to perform hyperparameter tuning by randomly searching a defined grid of parameters.
- **GPU Acceleration**: The model training process is accelerated using CUDA-enabled GPUs, making it efficient for large datasets.
- **Cross-Validation**: 5-fold cross-validation is used to validate the model performance during the hyperparameter search.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/xgboost-car-price-prediction.git
   cd xgboost-car-price-prediction
   ```

2. Install the necessary Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook to train the model and perform hyperparameter tuning.

## Model Training

The model is trained using XGBoost's `XGBRegressor`, and the following features are used:

- **Numerical features**: `milage`, `model_year`, `horsepower`, `cylinders`, `engine_size`
- **Log-transformed target**: The target variable (`price`) is log-transformed to handle skewness in the price distribution.

## Hyperparameter Tuning

The hyperparameters of the XGBoost model are tuned using `RandomizedSearchCV` with GPU acceleration. The following parameters are tuned:

- `lambda`: L2 regularization term on weights.
- `alpha`: L1 regularization term on weights.
- `colsample_bytree`: Fraction of features to be used for each tree.
- `subsample`: Fraction of data to be used for each tree.
- `learning_rate`: Step size shrinkage.
- `max_depth`: Maximum depth of trees.
- `min_child_weight`: Minimum sum of instance weight needed in a child.
- `n_estimators`: Number of boosting rounds.

## Results

The best hyperparameters found by RandomizedSearchCV are used to fit the final model, and predictions are made on the test data. 

Performance is measured using **Root Mean Squared Error (RMSE)**, with the results being printed during each iteration of the hyperparameter tuning process.

### Example Output:

```bash
Best hyperparameters found: {'max_depth': 15, 'lambda': 0.05, 'alpha': 0.05, ...}
Cross-validated RMSE: 1500.25
```

## License

This project is licensed under the MIT License.
