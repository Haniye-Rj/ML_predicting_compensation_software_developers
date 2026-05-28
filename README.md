# ML_predicting_compensation_software_developers
## Project Overview

This project builds a machine learning pipeline to predict developers’ annual compensation in USD using survey-style demographic, professional, technical, and workplace features.

The target variable is `annual_pay_usd`. Because salary data is highly skewed and contains extreme outliers, the project uses a log-transformed target (`log_annual_pay`) during model training and converts predictions back to real salary values for interpretation and final output.

The workflow includes exploratory data analysis, feature engineering, statistical feature validation, multicollinearity checks, model selection, hyperparameter tuning, and final prediction generation for a test dataset.

## Business Problem

Developer salary is influenced by many interacting factors, including region, work arrangement, education, job role, company size, technical stack, AI usage, decision-making power, and professional experience.

The goal of this project is to build a robust predictive model that can estimate annual developer pay while also identifying which professional and technical characteristics appear to be meaningful salary drivers.

## Dataset

The project uses two CSV files:

- `train.csv`: training data containing developer profile information and the salary target.
- `test.csv`: unseen data used to generate final salary predictions.

The original dataset includes numerical, categorical, boolean, and multi-select text features.

## Target Variable

The prediction target is:

```python
annual_pay_usd
```
<ul>
  <li><a href="my_plot.html">corrolation matrix</a></li>
During modeling, the target is transformed using:

```python
log_annual_pay = log1p(annual_pay_usd)
```

This transformation helps reduce the effect of extreme salaries and makes the distribution more suitable for regression models.

Predictions are converted back to USD using:

```python
predicted_salary_usd = expm1(predicted_log_salary)
```

## Exploratory Data Analysis

The notebook performs detailed EDA before modeling, including:

- Basic dataset inspection with `info()`, `describe()`, missing-value checks, and column review.
- Correlation analysis between numerical variables and salary.
- Salary distribution visualization before and after log transformation.
- Median salary comparison across categorical groups.
- Region-based salary analysis using grouped rare categories.
- Education-level salary analysis.
- Developer-role salary analysis.
- Work-location salary analysis.
- Operating-system salary analysis.
- Technology-purchase influence salary analysis.
- Programming-language-group salary analysis.
- Development-environment salary analysis.

Median salary is used for many comparisons instead of the mean because salary data contains large outliers.

## Impressive Analysis Worth Highlighting

### 1. Outlier-aware salary analysis

The project correctly recognizes that salary data is highly skewed and contains extreme values. Instead of relying only on average salary, it uses median salary in EDA and applies log transformation before modeling. This is a strong analytical decision because it reduces distortion from very high salaries.

### 2. Rare category grouping

Rare regions and age groups are grouped into `Other` categories. This reduces noise, prevents overfitting to very small groups, and makes visualizations and model features more stable.

### 3. Multicollinearity awareness

The notebook identifies a strong relationship between `coding_years_total` and `coding_years_professional`. Since these features carry overlapping information, `coding_years_total` is dropped to reduce redundancy. This shows an understanding of feature correlation and model stability.

### 4. Leakage prevention

The project drops features such as `job_satisfaction`, `age_group`, and `experience_years` due to leakage or redundancy concerns. This is important because a model should not rely on variables that may indirectly reveal the target or make the model unrealistic in production.

### 5. Domain-based feature engineering

Several raw categorical and multi-select columns are transformed into more meaningful model features:

- Developer roles are grouped into categories such as `Data & AI`, `Backend`, `Frontend`, `DevOps & Cloud`, `Leadership & Management`, and `QA & Testing`.
- Education is grouped into interpretable education levels.
- Company size is converted into an ordinal scale from small/freelance to enterprise.
- Programming languages are grouped into frontend, backend, systems, data, and scripting/devops categories.
- Development environments are grouped into IDE/tooling families.
- Technology-purchase influence is converted into a binary indicator.

This makes the model more interpretable and reduces unnecessary dimensionality.

### 6. Statistical feature validation

The project uses both One-Way ANOVA and Kruskal-Wallis tests to check whether categorical features show statistically significant salary differences. Using both tests is impressive because ANOVA is parametric, while Kruskal-Wallis is non-parametric and more robust when assumptions are not perfectly met.

### 7. Correlation check on engineered binary features

After creating binary and dummy variables, the project calculates correlations between engineered features and checks for high correlations above 0.70. This helps identify redundant predictors and supports cleaner feature selection.

### 8. Robust preprocessing pipeline

The modeling stage uses a `ColumnTransformer` with:

- `RobustScaler` for numerical features.
- `OneHotEncoder(handle_unknown='ignore')` for categorical features.
- Boolean passthrough for binary features.

This is production-minded because the pipeline can handle mixed data types and unseen categories in test data.

### 9. Model comparison and hyperparameter tuning

The project compares multiple regression approaches:

- Linear Regression
- Support Vector Regression
- K-Nearest Neighbors Regression

It uses `GridSearchCV` with 5-fold cross-validation and RMSE scoring to tune feature selection and model hyperparameters. This is more reliable than choosing a model based on a single train-test split.

### 10. End-to-end prediction workflow

The notebook applies the same transformations to the test set and generates final salary predictions. This shows that the project is not only exploratory but also structured as a complete machine learning workflow from raw data to prediction output.

## Feature Engineering Summary

Key engineered features include:

| Feature | Description |
|---|---|
| `log_annual_pay` | Log-transformed salary target |
| `coding_years_log` | Log-transformed professional coding experience |
| `has_influence` | Binary indicator for strong technology-purchase influence |
| `uses_ai` | Binary AI usage indicator |
| `company_size` | Ordinal company size category |
| `frontend` | Uses frontend technologies |
| `backend` | Uses backend technologies |
| `systems` | Uses systems programming languages |
| `data` | Uses data-related languages |
| `scripting_devops` | Uses scripting or DevOps languages |
| `mobile_dev` | Uses mobile development environments |
| grouped `dev_role` | Simplified developer role category |
| grouped `education` | Simplified education level |
| grouped `work_os` | Operating system tier category |

## Features Removed

Several columns are removed to reduce noise, avoid leakage, or simplify modeling. Examples include:

- `job_satisfaction`
- `age_group`
- `experience_years`
- `coding_years_total`
- `is_dev_professional`
- raw multi-select technology columns
- low-signal learning-method flags
- redundant development-environment indicators

## Modeling Approach

The final modeling pipeline follows this structure:

1. Split features and target.
2. Log-transform the salary target.
3. Identify numerical, categorical, and boolean columns.
4. Apply preprocessing with `ColumnTransformer`.
5. Apply feature selection using `SelectKBest(f_regression)`.
6. Train and tune multiple regression models.
7. Evaluate models using salary-scale metrics.
8. Generate predictions on the test set.

## Models Used

### Linear Regression

Used as a baseline interpretable model. Hyperparameter tuning focuses on the number of selected features.

### Support Vector Regression

Used as a stronger nonlinear model. The tuned parameters include:

- `C`
- `epsilon`
- `gamma`
- `kernel`
- number of selected features

### K-Nearest Neighbors Regression

Used as a distance-based comparison model. The tuned parameters include:

- number of neighbors
- distance metric
- weighting strategy
- number of selected features

## Evaluation Metrics

The project evaluates predictions after converting them back from log salary to real USD salary.

Metrics used:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R² Score

The notebook contains code to print the best cross-validation RMSE and real salary metrics for each model. The saved notebook does not currently contain executed output values for those final metric cells, so the exact numbers should be added after rerunning the notebook.

## Final Prediction Output

The final model generates a prediction column:

```python
predicted_salary_usd
```

and saves prediction results to CSV files, including:

```text
linear_salary_predictions.csv
predicted_salary_results.csv
```

## Project Strengths

- Strong handling of skewed target data through log transformation.
- Careful outlier-aware EDA using median salary.
- Meaningful domain-based feature engineering.
- Attention to leakage and redundant variables.
- Statistical testing for feature relevance.
- Use of robust preprocessing for mixed data types.
- Cross-validation and hyperparameter tuning.
- Comparison of multiple regression algorithms.
- Complete workflow from raw data to prediction file.

## Possible Improvements

Future improvements could include:

- Add final saved model metrics directly to the notebook output.
- Package preprocessing into reusable functions to avoid repeated code.
- Use a single final pipeline for both training and inference.
- Compare additional models such as Random Forest, Gradient Boosting, XGBoost, LightGBM, or CatBoost.
- Add feature-importance analysis using permutation importance or SHAP.
- Save the trained model with `joblib` for reproducible deployment.
- Add a `requirements.txt` file.
- Add clearer section headings and markdown explanations throughout the notebook.

## How to Run

1. Install the required Python libraries:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn plotly statsmodels
```

2. Place the data files in the working directory:

```text
train.csv
test.csv
```

3. Open and run the notebook:

```text
ML (3).ipynb
```

4. Run all cells in order.

5. Review generated outputs:

```text
cleaned_data1.csv
linear_salary_predictions.csv
predicted_salary_results.csv
```

## Conclusion

This project demonstrates a complete machine learning workflow for salary prediction. The strongest parts of the project are the thoughtful preprocessing decisions, outlier-aware salary analysis, domain-driven feature engineering, statistical feature validation, and model comparison using cross-validation. These elements make the project more impressive than a simple regression notebook because they show both technical implementation and analytical reasoning.

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


