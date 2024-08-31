# Project Structure
- EDA & preprocessing: Exploratory Data Analysis and data preprocessing
- Model: Implementation of different models (Naive Bayes, Logistic Regression, XGBoost, Neural Network)
- Testing: Model evaluation and performance metrics
- Explainability: Feature importance analysis for different models

# Running the Code

## Data Exploration:
Run `EDA & preprocessing/EDA.ipynb` for data analysis

## Data Preprocessing:
Run `preprocessing/train_test_split.py`

## Model Training:

- Naive Bayes: `Model/nb.py`
- Logistic Regression: `Model/lr.py`
- XGBoost:
  - For local execution: `Model/xgb.py`
  - For SCC batch job, see: `Model/xgb.sh`
- Neural Network:
  - For local execution: `Model/nn.py`
  - For SCC batch job, see: `Model/nn.sh`

## Model Testing:

Run `Testing/testing.ipynb`

## Explainability Analysis:

Run `Explainability/explainability.ipynb`
