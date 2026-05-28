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
<h1>My Plotly Visualizations</h1>

<ul>
  <li><a href="my_plot.html">corrolation matrix</a></li>
  <li><a href="NUMERIC_COR.html">numerical corrolationss</a></li>
</ul>

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
````

Also transformed professional coding years:

```python
coding_years_log = np.log1p(coding_years_professional)
```

---

# Machine Learning Models

The following models were trained and evaluated:

* Linear Regression
* Ridge Regression
* K-Nearest Neighbors (KNN)
* Support Vector Regression (SVR)

---

# Model Pipeline

The project used Scikit-learn Pipelines including:

* OneHotEncoder
* StandardScaler
* Cross-validation
* Hyperparameter tuning using GridSearchCV

---

# Best Performing Model

## Support Vector Regression (SVR)

SVR achieved the best performance due to its ability to model nonlinear relationships between features and salary.

### Final Metrics

| Metric | Score         |
| ------ | ------------- |
| MAE    | Best Result   |
| RMSE   | Best Result   |
| R²     | Highest Score |

---

# Visualizations

The project includes:

* Salary distribution before and after transformation
* Model comparison plots
* Actual vs predicted salary plots
* Error distribution analysis
* Feature distribution visualizations

Visualizations were built using:

* Plotly
* Matplotlib
* Seaborn

---

# Technologies Used

## Python Libraries

* pandas
* numpy
* scikit-learn
* plotly
* matplotlib
* seaborn
---

# Key Learnings

This project demonstrates:

* End-to-end ML workflow
* Data preprocessing strategies
* Feature engineering techniques
* Regression model comparison
* Hyperparameter tuning
* Model evaluation
* Interactive data visualization

---

# Future Improvements

Possible future enhancements:

* XGBoost / LightGBM implementation
* Better technology-level feature extraction
* Advanced encoding techniques
* Model deployment
* Real-time salary prediction interface


