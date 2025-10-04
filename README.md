❤️ Heart Disease Prediction Project

📑 Table of Contents

Overview

Dataset

Features

Project Steps

Results

Usage

Optional Enhancements

🚀 Overview

This project predicts the presence of heart disease in patients using a structured dataset.
It leverages machine learning models to classify patients based on clinical features and provides interpretable results for healthcare decision support.
💡 Goal: Early prediction of heart disease to assist preventive healthcare.

📊 Dataset

Source: UCI Heart Disease Dataset (heart_disease_data.csv)

Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target

Target:

0 = No heart disease

1 = Presence of heart disease


✨ Features

Categorical: sex, cp, fbs, restecg, exang, slope, ca, thal

Numerical: age, trestbps, chol, thalach, oldpeak
💡 Understanding feature impact can guide doctors in preventive care.

🛠️ Project Steps
1️⃣ Exploratory Data Analysis (EDA)

Inspected dataset shape, missing values, and statistics

Visualized feature distributions with histograms, boxplots, and correlation heatmaps

Checked target balance


2️⃣ Data Preprocessing

Split features into categorical and numerical

Applied Standard Scaling for numeric features

Applied One-Hot Encoding for categorical features

Split dataset into train (80%) and test (20%)

3️⃣ Modeling

Trained three models: Logistic Regression, Random Forest, XGBoost

Evaluated using: Accuracy, Precision, Recall, F1-score, ROC-AUC


4️⃣ Hyperparameter Tuning

Optimized Random Forest using RandomizedSearchCV

Achieved best ROC-AUC: 0.926


5️⃣ Feature Importance

Top 15 features visualized using Random Forest importance scores


6️⃣ Model Saving

Saved the best Random Forest model as best_heart_disease_model.pkl for future predictions

📈 Results
Model	Accuracy	ROC-AUC
Logistic Regression	0.84	0.91
Random Forest	0.77	0.90
XGBoost	0.75	0.75