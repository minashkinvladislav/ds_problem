# Multi-Label Classification Task

## Problem Description

Solution for multi-label classification task to predict object membership in 14 different service categories (service_a - service_n). The task is formulated as 14 independent binary classification problems with LogLoss quality metric.

## Data Structure

- problem_train.csv: Training dataset with object features
- problem_labels.csv: Labels for 14 categories (0/1 for each category)
- problem_test.csv: Test dataset for prediction
- problem_test_labels.csv: Result - probabilities of belonging to each category

### Feature Types:
- Numerical features (116 features)
- Ordinal features (211 features)
- Categorical features (1051 features)
- Additional feature (release feature, one out of three types)

## Approach

### Feature Selection
Implemented feature selection algorithm based on Information Value (IV):

- **Data completeness filtering**: Exclude features with missing rate > 15%
- **Information Value calculation**: For each feature relative to target variable
- **IV threshold**: Select features with IV ≥ 0.1
- **Numerical feature binning**: Split into quantiles for IV calculation

### Data Preprocessing
- Missing value handling:
  - Categorical/ordinal features: fill with 'missing' value
  - Numerical features: fill with median
- Type conversion:
  - Ordinal features: convert to int (where possible)
  - Categorical: convert to string type

### Model
CatBoost Classifier for each of 14 categories:

```python
CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    cat_features=categorical_features,
    eval_metric='Logloss',
    early_stopping_rounds=50
)
```

### Validation
- Split training data into train/validation (80/20)
- Stratified split to preserve class proportions
- Early stopping based on validation Logloss

### Folder Structure

```
├── feature_selection.py    # Feature selection by Information Value
├── train_and_predict.py    # Main training and prediction pipeline
├── catboost_models/        # Saved models for each category
└── data/                   # Source data and results
```

## Key Features

- Separate model trained for each category
- Each category uses its own set of most informative features


## Usage

```bash
uv sync
python train_and_predict.py
```

Results are saved to `data/problem_test_labels.csv` with probabilities of belonging to each of the 14 categories.

## Implementation Details

### Feature Selection Process
- Calculate Information Value for each feature type separately
- Apply different binning strategies for numerical vs categorical features
- Filter features based on both statistical significance and data quality

### Model Training Strategy
- Train independent CatBoost models for each target label
- Use stratified validation to handle class imbalance
- Apply early stopping to prevent overfitting
- Save trained models for reproducible predictions

### Prediction Pipeline
- Load saved models for each category
- Apply same preprocessing as during training
- Generate probability predictions for test set
- Combine predictions into final submission format
