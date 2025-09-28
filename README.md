# Heart Disease Prediction Project

## Overview
This project aims to **predict the presence of heart disease** in patients using a structured dataset. It leverages machine learning models to classify patients based on clinical features and provides interpretable results for healthcare decision support.

---

## Dataset
- **Source:** UCI Heart Disease dataset (or your own CSV: `heart_disease_data.csv`)  
- **Columns:**
  - `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`  
- **Target:** `0` = no heart disease, `1` = presence of heart disease  

---

## Features
- **Categorical:** `sex, cp, fbs, restecg, exang, slope, ca, thal`  
- **Numerical:** `age, trestbps, chol, thalach, oldpeak`  

---

## Project Steps

1. **Exploratory Data Analysis (EDA)**
   - Inspected dataset shape, missing values, and statistics  
   - Visualized distributions with histograms and correlation heatmaps  
   - Checked target balance  

2. **Data Preprocessing**
   - Defined categorical and numerical columns  
   - Applied **Standard Scaling** for numeric features  
   - Applied **One-Hot Encoding** for categorical features  
   - Split dataset into training and test sets (80/20)  

3. **Modeling**
   - Trained three models:
     - Logistic Regression
     - Random Forest Classifier
     - XGBoost Classifier  
   - Evaluated using **accuracy, precision, recall, F1-score, ROC-AUC**  

4. **Hyperparameter Tuning**
   - Optimized Random Forest using **RandomizedSearchCV**  
   - Achieved **best ROC-AUC: 0.926**  

5. **Feature Importance**
   - Extracted and visualized top 15 features using Random Forest  

6. **Model Saving**
   - Saved the **best Random Forest model** as `best_heart_disease_model.pkl` for later predictions  

---

## Results

| Model               | Accuracy | ROC-AUC |
|--------------------|----------|---------|
| Logistic Regression | 0.84     | 0.91    |
| Random Forest       | 0.77     | 0.90    |
| XGBoost             | 0.75     | 0.75    |

- Logistic Regression gives the **best overall performance**.  
- Random Forest is highly sensitive to detecting heart disease (high recall for class 1).  

---

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Heart_Disease_Prediction.git
