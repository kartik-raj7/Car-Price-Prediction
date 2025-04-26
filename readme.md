# 🚗 Car Price Prediction - Machine Learning Project

## 📋 Project Overview
This project aims to **predict car prices** based on various features such as engine size, curb weight, horsepower, and other specifications.  
The objective is to help businesses **estimate car prices accurately** using machine learning techniques.

---

## 🎯 Problem Statement
Manually estimating car prices can be inconsistent due to the variety of influencing factors.  
Our goal was to build a **regression model** that predicts car prices reliably based on available data.

---

## 📊 Dataset
- Dataset includes details like brand, fuel type, engine size, body type, etc.
- Key preprocessing steps:
  - Handling missing values
  - Encoding categorical variables (Label Encoding and Target Encoding)
  - Outlier detection and removal
  - Feature selection using SelectKBest, RFE, Lasso, and Random Forest

---

## 🔨 Project Workflow

- **Data Cleaning**: Removed unnecessary columns and cleaned missing data.
- **Exploratory Data Analysis (EDA)**:
  - Visualized feature distributions
  - Studied relationships (e.g., car length vs price, engine size vs price)
  - Built a correlation matrix to find influential features
- **Feature Engineering**:
  - Applied Label Encoding and Target Encoding
- **Outlier Handling**:
  - Identified using boxplots; extreme outliers were removed
- **Feature Selection**:
  - Selected top features highly correlated with car prices
- **Model Training**:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest Regressor
  - XGBoost Regressor
  - Support Vector Regressor (SVR)
- **Model Tuning**:
  - K-Fold Cross Validation
  - Hyperparameter tuning (Grid Search, Gradient Descent optimization)

---

## 🏆 Results

| Model | R² Score | MAE | RMSE |
|:---|:---|:---|:---|
| Ridge Regression | 0.7516 | 1239.88 (12.46%) | 1823.62 (18.33%) |
| Lasso Regression | 0.7516 | 1239.88 (12.46%) | 1823.65 (18.33%) |
| Random Forest (Simple) | 0.8950 | 851.01 (8.56%) | 1185.61 (11.92%) |
| Random Forest (K-Fold CV) | 0.8548 | 1004.64 (9.57%) | 1596.72 (15.21%) |
| XGBoost | 0.7552 | 1590.62 (14.65%) | 2314.40 (21.32%) |
| SVR | -0.0086 | 3030.39 (30.46%) | 3674.58 (36.94%) |

✅ **Best model**: Random Forest Regressor (Simple)

---

## 📈 Final Observations

- **For most normal cars**, the model predicts prices **very accurately**.
- **For cars with extreme specs** (very large engines, very light weights), the model makes **larger errors**.
- **Recommendation**:  
  For businesses dealing with exotic cars, it would help to collect more data and possibly develop a specialized model.

---

## 🚀 Future Improvements

- Collect more data for extreme specification cars.
- Explore deep learning models for non-linear feature capture.
- Deploy the model via a web application using Streamlit or Flask.

---

## 📚 Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - streamlit

---

## 🙋‍♂️ About Me
I'm an aspiring machine learning engineer passionate about solving real-world business problems using data-driven solutions.  
---
