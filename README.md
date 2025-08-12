# Telco Customer Churn Prediction

## Overview
This project predicts whether a customer will churn using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset).  
The workflow covers data preprocessing, class balancing, feature selection (SelectKBest), dimensionality reduction (PCA), and training multiple machine learning models.

## Project Workflow
1. **Data Exploration**  
   - Examined dataset structure, missing values, and feature correlations.  
   - Found that "Churn Value" is most correlated with "Churn Score" (0.66).

2. **Feature Engineering & Class Balancing**  
   - Encoded categorical variables and balanced classes using oversampling.

3. **Feature Selection**  
   - Applied SelectKBest to identify top predictive features.  
   - Trained models: KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost.  
   - Used GridSearchCV for hyperparameter tuning.  

4. **Dimensionality Reduction**  
   - Applied PCA and repeated model training and tuning.

5. **Model Comparison & Findings**  
   - Random Forest achieved the highest accuracy for both SelectKBest and PCA.  
   - KNN and Decision Tree had the lowest accuracy.  
   - Precision for "No Churn" (~0.90s) was higher than for "Churn" (~0.70s).

## Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn, XGBoost  
- Matplotlib, Seaborn  

## Dataset
Telco Customer Churn dataset from Kaggle:
https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset
