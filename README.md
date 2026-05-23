# ML_predicting_compensation_software_developers
 to build a model predicting the annual compensation of software developers based on the training sample and generate predictions for all observations from the test sample.
# Developer Salary Prediction using Machine Learning

## Overview

This project predicts annual developer salaries using machine learning techniques based on developer survey data.  
The workflow includes:

- Data cleaning and preprocessing
- Feature engineering
- Exploratory data analysis (EDA)
- Feature transformation
- Model training and evaluation
- Hyperparameter tuning
- Final salary prediction on unseen data
- Interactive visualization using Plotly and Streamlit

The final selected model was **Support Vector Regression (SVR)**, which achieved the best performance compared to other regression models.

---

# Project Goals

The objective of this project is to:

- Predict developer salaries from survey features
- Explore the impact of experience, technologies, education, and work conditions on salary
- Compare multiple regression algorithms
- Build a reproducible end-to-end machine learning pipeline

---

# Dataset

The dataset contains developer-related information such as:

- Region
- Employment type
- Education
- Developer role
- Company size
- Operating systems
- AI usage
- Programming languages
- Coding experience

Target variable:

- `annual_pay_usd`

---

# Data Preprocessing

The following preprocessing steps were applied:

## Cleaning

- Removed irrelevant columns
- Handled missing values
- Standardized column names
- Removed invalid salary values

## Feature Engineering

Created engineered features such as:

- `frontend`
- `backend`
- `systems`
- `data`
- `scripting_devops`
- `mobile_dev`

Grouped developer roles into broader categories.

## Target Transformation

Salary distribution was highly right-skewed.

Applied logarithmic transformation:

```python
log_annual_pay = np.log1p(annual_pay_usd)
