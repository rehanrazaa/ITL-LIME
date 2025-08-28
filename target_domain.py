import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import scipy.stats as stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
import plotly.express as px
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
# Ignore all warnings
import joblib
warnings.filterwarnings('ignore')
from sklearn.svm import SVC


def get_target_data():
    # Step 1: Read the target dataset /media/linuxdata/rr35021174/ITL-Github/Dataset
    target_data = pd.read_csv('/media/linuxdata/rr35021174/ITL-Github/Dataset/target_set_1_east.csv')  # Replace with the correct path
    # Step 2: Handle invalid entries like '?' by replacing them with NaN
    target_data.replace('?', pd.NA, inplace=True)

    # Step 3: Label Encoding for categorical features in target domain
    label_encoder = LabelEncoder()

    # Categorical columns in the target dataset (same as source dataset)
    categorical_columns_target = ['Gender', 'City', 'Profession', 'Degree', 'Dietary Habits',
                                    'Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']

    # Apply Label Encoding for each categorical column in the target dataset
    for col in categorical_columns_target:
        target_data[col] = label_encoder.fit_transform(target_data[col])

    # Step 4: Handle 'Sleep Duration' in target domain (convert categorical to numeric)
    sleep_mapping = {
            "'Less than 5 hours'": 4,
            "'5-6 hours'": 5.5,
            "'7-8 hours'": 7.5
    }
    target_data['Sleep Duration'] = target_data['Sleep Duration'].map(sleep_mapping)

    # Step 5: Convert columns to numeric (to handle any non-numeric values in numerical columns)
    # Convert columns to numeric (force any non-numeric values to NaN)
    numerical_columns_target = ['Age', 'CGPA', 'Work/Study Hours', 'Financial Stress', 'Study Satisfaction',
                                    'Job Satisfaction', 'Sleep Duration', 'Work Pressure', 'Academic Pressure']

    target_data[numerical_columns_target] = target_data[numerical_columns_target].apply(pd.to_numeric, errors='coerce')

    # Step 6: Impute missing values in numerical columns
    imputer = SimpleImputer(strategy='mean')  # Impute missing values with the mean
    target_data[numerical_columns_target] = imputer.fit_transform(target_data[numerical_columns_target])

    # Step 7: Standardize numerical features (same as source domain)
    scaler_target = StandardScaler()

    # Apply standardization to numerical columns in target data
    target_data[numerical_columns_target] = scaler_target.fit_transform(target_data[numerical_columns_target])

    # Step 8: Define the features (X) and target (y) for the target domain
    X_target = target_data.drop(columns=['Depression'])  # Features (all columns except 'Depression')
    y_target = target_data['Depression']  # Target variable ('Depression')

    # Step 9: Split the target data into training and testing sets
    X_train_target, X_test_target, y_train_target, y_test_target = train_test_split(X_target, y_target, test_size=0.2, random_state=42)

    # Step 10: Display the shapes of the resulting datasets for target domain
    # Step 11: Drop rows with missing values in the target domain
    X_train_target = X_train_target.dropna()
    y_train_target = y_train_target[X_train_target.index]  # Align target variable

    X_target_test = X_test_target.dropna()
    y_target_test = y_test_target[X_test_target.index]  # Align target variable
        
    rf_model_target = joblib.load('/media/linuxdata/rr35021174/ITL-Github/Trained_BB_models/final_gsvm_model_target_set_1_east.pkl')
    # Step 9: Make predictions on the test set in the target domain
    #y_pred_target = rf_model_target.predict(X_target_test)
    #y_prob_target = rf_model_target.predict_proba(X_target_test)[:, 1]  # Probabilities for AUC (positive class)

    # Step 10: Evaluate the model's performance on the target data
    # accuracy_target = accuracy_score(y_target_test, y_pred_target)
    # precision_target = precision_score(y_target_test, y_pred_target)
    # recall_target = recall_score(y_target_test, y_pred_target)
    # f1_target = f1_score(y_target_test, y_pred_target)
    # auc_target = roc_auc_score(y_target_test, y_prob_target)

    # # Step 11: Print the metrics for the target domain
    # print(f"Accuracy: {accuracy_target:.4f}")
    # print(f"Precision: {precision_target:.4f}")
    # print(f"Recall: {recall_target:.4f}")
    # print(f"F1-Score: {f1_target:.4f}")
    # print(f"AUC: {auc_target:.4f}")

    
    return X_target, y_target, X_train_target, X_target_test, y_train_target, y_target_test, rf_model_target

#get_target_data()

