# Churn Analysis – Customer Churn Prediction using Machine Learning

Machine learning-based churn prediction using the Telco Customer Churn dataset. This project focuses on predicting customer churn for a telecom subscription service by building an end-to-end machine learning pipeline. It covers data preprocessing, exploratory data analysis (EDA), handling class imbalance using SMOTE, training multiple classifiers (like Decision Tree, Random Forest, and XGBoost), and implementing deployment-ready prediction logic.

## Project Objective

To build a machine learning model that can predict whether a customer will churn (i.e., stop using the service) based on various features like tenure, service usage, contract type, and payment method.

## Dataset

- **Source**: Telco Customer Churn dataset
- **Records**: 7043
- **Features**: 21 (including both numerical and categorical)

## Key Steps

1. **Data Cleaning**  
   - Removed unnecessary columns like `customerID`
   - Handled missing or incorrect data in `TotalCharges`
   - Converted data types appropriately

2. **Exploratory Data Analysis (EDA)**  
   - Distribution plots, box plots, and correlation heatmaps for numerical features
   - Count plots for categorical variables
   - Customized plot colors for better clarity

3. **Preprocessing**  
   - Label encoding of all categorical variables
   - SMOTE applied to balance the target classes

4. **Model Training**  
   - Compared Decision Tree, Random Forest, and XGBoost classifiers
   - 5-fold cross-validation used for accuracy estimation
   - Random Forest chosen as final model (accuracy ~77.8%)

5. **Model Deployment**  
   - Trained model saved with pickle
   - Prediction system created for custom input

## Model Performance

- **Final Model**: Random Forest Classifier
- **Test Accuracy**: ~77.8%
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

## Future Improvements

- Apply hyperparameter tuning (e.g., GridSearchCV)
- Try more advanced models (e.g., LightGBM, CatBoost)
- Experiment with feature selection or engineering
- Build a web UI using Streamlit or Flask

## Requirements

To run this project, install the following Python libraries:

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- xgboost  
- imbalanced-learn  

You can install them all at once with:  
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
   

