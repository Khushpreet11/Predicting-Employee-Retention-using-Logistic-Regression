# 🧑‍💼 Predicting Employee Retention

A machine learning project that predicts employee attrition using **Logistic Regression** and **Random Forest**, built as part of a data science assignment. The goal is to help organizations proactively identify employees at risk of leaving and enable timely HR interventions.

---

## 📌 Table of Contents

- [Business Objective](#business-objective)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Project Workflow](#project-workflow)
- [Modeling & Results](#modeling--results)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Recommendations & Next Steps](#recommendations--next-steps)
- [Authors](#authors)

---

## Business Objective

A mid-sized technology company wants to improve its understanding of employee retention and proactively identify employees likely to leave. This project builds a predictive model to classify whether an employee will **Stay** or **Leave**, using historical HR data — enabling the organization to focus retention efforts where they matter most.

---

## Dataset

**File:** `Employee_data.csv`

| Property | Value |
|---|---|
| Rows | 74,610 |
| Columns | 24 |
| Target Variable | `Attrition` (Stayed / Left) |

**Feature Overview:**

| Column | Description |
|---|---|
| Employee ID | Unique identifier |
| Age | Age of the employee |
| Gender | Male / Female |
| Years at Company | Total years with the company |
| Job Role | Employee's job function |
| Monthly Income | Monthly salary |
| Work-Life Balance | Rated scale |
| Job Satisfaction | Satisfaction score |
| Performance Rating | Annual performance rating |
| Number of Promotions | Count of promotions received |
| Overtime | Whether employee works overtime |
| Distance from Home | Commute distance (km) |
| Education Level | Highest qualification |
| Marital Status | Single / Married / Divorced |
| Number of Dependents | Number of dependents |
| Job Level | Entry / Mid / Senior |
| Company Size | Small / Medium / Large |
| Company Tenure (In Months) | Tenure in months |
| Remote Work | Whether employee works remotely |
| Leadership Opportunities | Access to leadership roles |
| Innovation Opportunities | Access to innovation projects |
| Company Reputation | Employee perception of company brand |
| Employee Recognition | Recognition received |
| **Attrition** | **Target: Stayed / Left** |

---

## Exploratory Data Analysis

### 🎯 Attrition Distribution

The dataset is fairly balanced — roughly 52% of employees stayed and 48% left.

![Attrition Distribution](images/01_attrition_distribution.png)

---

### 💼 Attrition Rate by Job Role

Attrition rates are relatively consistent across roles, with **Finance** and **Technology** showing slightly higher churn.

![Attrition by Job Role](images/02_attrition_by_role.png)

---

### 💰 Monthly Income vs. Attrition

Employees who left tend to cluster at **lower income brackets**, suggesting salary is a meaningful retention factor.

![Income Distribution by Attrition](images/03_income_by_attrition.png)

---

### 💍 Attrition by Marital Status & Overtime

**Single employees** and those working **overtime** show noticeably higher attrition rates.

![Marital Status and Overtime](images/04_marital_overtime.png)

---

### 🎂 Age Distribution by Attrition

Younger employees (20–35) have a higher tendency to leave compared to mid-career and senior employees.

![Age Distribution by Attrition](images/06_age_by_attrition.png)

---

### 🏆 Top Feature Importances (Random Forest)

Distance from Home, Monthly Income, and Company Tenure are the strongest predictors of attrition.

![Feature Importance](images/05_feature_importance.png)

---

## Project Workflow

```
1. Data Understanding       → Load data, inspect shape, types, basic statistics
2. Data Cleaning            → Handle missing values, remove redundant columns
3. Train–Validation Split   → 80/20 stratified split
4. EDA (Training Data)      → Univariate, bivariate, correlation, class balance
5. Feature Engineering      → One-hot encoding of categoricals, StandardScaler
6. Model Building           → RFE for feature selection, Logistic Regression (statsmodels)
7. Threshold Optimization   → ROC curve, Sensitivity–Specificity tradeoff, Precision–Recall curve
8. Model Evaluation         → Accuracy, Confusion Matrix, Sensitivity, Specificity, AUC
```

---

## Modeling & Results

### Logistic Regression (Primary Model)

- Feature selection via **Recursive Feature Elimination (RFE)** — 15 features selected
- Model built using **statsmodels** (with p-values and VIF analysis)
- Threshold tuned using ROC and Precision-Recall tradeoffs

| Metric | Train | Validation |
|---|---|---|
| Accuracy | ~75% | **75.26%** |
| AUC-ROC | — | **0.8425** |

### Random Forest (Benchmark — 200 Trees)

| Metric | Test |
|---|---|
| Accuracy | 75.00% |
| AUC-ROC | 0.8380 |

> Logistic Regression slightly outperforms Random Forest on both accuracy and AUC, while remaining highly interpretable.

---

## Key Features

**Top predictors of attrition (Random Forest importance):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | Distance from Home | 0.118 |
| 2 | Employee ID | 0.105 |
| 3 | Monthly Income | 0.098 |
| 4 | Company Tenure | 0.091 |
| 5 | Years at Company | 0.087 |
| 6 | Age | 0.082 |
| 7 | Marital Status — Single | 0.061 |
| 8 | Job Level — Senior | 0.055 |
| 9 | Job Level — Entry | 0.051 |
| 10 | Number of Dependents | 0.048 |

---

## Tech Stack

- **Python 3**
- `pandas`, `numpy` — Data manipulation
- `matplotlib`, `seaborn` — Visualization
- `scikit-learn` — Train/test split, RFE, StandardScaler, Random Forest, metrics
- `statsmodels` — Logistic Regression with statistical summaries

---

## Project Structure

```
├── Employee_data.csv                          # Raw dataset
├── Predicting_Employee_Retention_1.ipynb      # Main analysis notebook
├── Predicting_Employee_Retention_Report.pdf   # Holistic summary report
├── images/                                    # Visualizations for README
│   ├── 01_attrition_distribution.png
│   ├── 02_attrition_by_role.png
│   ├── 03_income_by_attrition.png
│   ├── 04_marital_overtime.png
│   ├── 05_feature_importance.png
│   └── 06_age_by_attrition.png
└── README.md
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Khushpreet11/Logistic-Regression.git
   cd Logistic-Regression
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook Predicting_Employee_Retention_1.ipynb
   ```

4. **Update the data path** in the notebook if needed:
   ```python
   DATA_PATH = "Employee_data.csv"
   ```

---

## Recommendations & Next Steps

- **Hyperparameter Tuning** — Apply `GridSearchCV` to optimize model parameters
- **Cross-Validation** — Use k-fold CV for more robust performance estimates
- **Class Imbalance Handling** — Explore SMOTE or `class_weight='balanced'`
- **Model Explainability** — Implement SHAP or LIME for interpretable predictions with stakeholders
- **Deployment** — Deploy as an HR monitoring tool to flag at-risk employees in real time

---

## Authors

- **Khushpreet**
- **Rosy Samantaray**
- **Prity**
- **Raaz**
