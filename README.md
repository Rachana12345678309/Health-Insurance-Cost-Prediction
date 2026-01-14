# Health Insurance Cost Prediction using Linear Regression

A machine learning project designed to predict individual medical insurance costs by analyzing various factors such as age, BMI, smoking status, and geographical region using a Multiple Linear Regression model.

## Overview

This project demonstrates a standard end-to-end data science workflow: from data cleaning and exploratory analysis to feature engineering and predictive modeling. The primary goal is to build a robust Linear Regression model that can accurately estimate `charges` (the insurance premium) based on health and demographic features while ensuring the model satisfies statistical assumptions like the absence of multicollinearity.

## Dataset

- **Source:** Insurance dataset (`17_insurance.csv`).
- **Key columns:**
  - `age`: Age of the primary beneficiary.
  - `sex`: Insurance contractor gender (female, male).
  - `bmi`: Body mass index, providing an understanding of body weight relative to height.
  - `children`: Number of children covered by health insurance / Number of dependents.
  - `smoker`: Smoking status (yes, no).
  - `region`: The beneficiary's residential area in the US (northeast, southeast, southwest, northwest).
  - `charges`: Individual medical costs billed by health insurance (Target Variable).
  - *Note: The dataset also includes extended features like `past_consultations`, `num_of_steps`, and `Anual_Salary` for deeper analysis.*

## Objectives

- Perform **Exploratory Data Analysis (EDA)** to understand the correlation between features and insurance costs.
- Handle **Categorical Encoding** for variables like `sex`, `smoker`, and `region`.
- Address **Multicollinearity** by calculating the **Variance Inflation Factor (VIF)**.
- Build and optimize a **Multiple Linear Regression** model.
- Evaluate model performance using metrics such as **R-squared ($R^2$)**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**.

## Methods and Analysis

The notebook follows a detailed step-by-step approach:

- **Data Preprocessing**
  - Handling missing values and duplicates.
  - Visualizing distributions using histograms and boxplots to detect outliers.
  - Label encoding for binary and multi-class categorical variables.

- **Feature Selection & Statistical Validation**
  - **Correlation Matrix:** Identifying features with the strongest relationship to `charges`.
  - **VIF Analysis:** Systematically removing features with high multicollinearity (VIF > 5) to ensure stable model coefficients.
  - **Train-Test Split:** Splitting data (80/20) to validate model performance on unseen data.

- **Model Building and Evaluation**
  - Implementing `LinearRegression` from Scikit-Learn.
  - **Performance Metrics:** - Achieved high $R^2$ scores, indicating a strong fit.
    - Calculated `MSE` and `RMSE` to quantify prediction error in dollar amounts.

- **Prediction Interface**
  - Includes a sample prediction block where new customer data can be fed into the model to generate an estimated insurance premium.

## Tech Stack

- **Language:** Python 3
- **Libraries:**
  - `pandas` and `numpy`: Data manipulation.
  - `seaborn` and `matplotlib`: Advanced data visualization.
  - `scikit-learn`: Model building, data splitting, and evaluation.
  - `statsmodels`: For calculating Variance Inflation Factor (VIF).
- **Environment:** Jupyter / Google Colab

## How to Run

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/insurance-cost-prediction.git
   cd insurance-cost-prediction

2. *Create and activate a virtual environment (optional but recommended):*
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. *Install dependencies:*
   pip install pandas numpy seaborn matplotlib scikit-learn statsmodels

4. Ensure the dataset is present: Place 17_insurance.csv in the root folder.

5. *Open the notebook:*
   jupyter notebook 17_LinearRegression_Handson_vgood.ipynb
