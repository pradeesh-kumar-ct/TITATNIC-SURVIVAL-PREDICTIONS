# Titanic Survival Prediction 🚢

Predicting passenger survival on the Titanic using machine learning.  
This project applies **feature engineering** (titles, family size, alone indicator) and trains a **RandomForestClassifier** with hyperparameter tuning via GridSearchCV.




## ⚙️ Features
- **Feature Engineering**: Extracted titles, family size, and alone indicator.
- **Preprocessing Pipelines**: Imputation, scaling, and one‑hot encoding.
- **Modeling**: RandomForestClassifier with cross‑validation.
- **Hyperparameter Tuning**: GridSearchCV for best parameters.
- **Submission File**: Generates `submission.csv` for Kaggle.
- ## 📊 Exploratory Data Analysis

### Survival by Gender
![Survival by Gender](plots/survival_by_gender.png)

### Age  Survival by gender 
![Age Distribution](plots/age_survival.png)


### Insight
- Females had a much higher survival rate (~74%) compared to males (~19%).
- Gender was a strong predictor of survival.


## 📊 Results
- Cross‑validation accuracy: ~82%  
- Best parameters (example):  
  ```python
  {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 2}