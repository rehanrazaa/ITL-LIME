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
# Ignore all warnings
import joblib
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
sns.set(style="whitegrid")

def get_source_data():
    # Load dataset
    source_data = pd.read_csv("Path/source_dataset_west.csv")
    # Label Encoding for categorical features
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # List of categorical columns to apply Label Encoding
    categorical_columns = ['Gender', 'City', 'Profession', 'Degree', 'Dietary Habits',
                        'Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']

    # Apply Label Encoding to each categorical column
    for col in categorical_columns:
        source_data[col] = label_encoder.fit_transform(source_data[col])

    # Handle 'Sleep Duration' (convert categorical to numeric)
    # Map the 'Sleep Duration' column to numeric values
    sleep_mapping = {
        "'Less than 5 hours'": 4,
        "'5-6 hours'": 5.5,
        "'7-8 hours'": 7.5
    }
    source_data['Sleep Duration'] = source_data['Sleep Duration'].map(sleep_mapping)

    # Standardize Numerical Features (Z-score normalization)
    # List of numerical columns that need to be standardized
    numerical_columns = ['Age', 'CGPA', 'Work/Study Hours', 'Financial Stress', 'Study Satisfaction',
                        'Job Satisfaction', 'Sleep Duration', 'Work Pressure', 'Academic Pressure']

    # Initialize StandardScaler for standardization
    scaler = StandardScaler()
    # Apply standardization to the numerical columns
    source_data[numerical_columns] = scaler.fit_transform(source_data[numerical_columns])
    categorical_columns = ['Gender', 'City', 'Profession', 'Degree', 'Dietary Habits',
                        'Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']
    numerical_columns = ['Age', 'CGPA', 'Work/Study Hours', 'Financial Stress', 'Study Satisfaction',
                        'Job Satisfaction', 'Sleep Duration', 'Work Pressure', 'Academic Pressure']


    # Define the features (X) and target (y)
    X = source_data.drop(columns=['Depression'])  # Features (all columns except 'Depression')
    y = source_data['Depression']  # Target (Depression)

    # Drop missing values
    X = X.dropna()  # Drop rows with missing values in features
    y = y[X.index]  # Align the target variable to the remaining rows in X

    # Split the data into training and testing sets (80% train, 20% test)
    X_train_source, X_test_source, y_train_source, y_test_source = train_test_split(X, y, test_size=0.2, random_state=42)

    # Black-Box Model - Initialize and train the Gaussian SVM (RBF kernel)
    # svm_model_source = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)
    # # Train the model
    # svm_model_source.fit(X_train_source, y_train_source)
    
    # # Step 2: Make predictions on the test set
    # y_pred_source = svm_model_source.predict(X_test_source)
    # y_prob_source = svm_model_source.predict_proba(X_test_source)[:, 1]  # Probabilities for AUC (positive class)

    # # Step 3: Evaluate the model's performance
    # accuracy_source = accuracy_score(y_test_source, y_pred_source)
    # precision_source = precision_score(y_test_source, y_pred_source)
    # recall_source = recall_score(y_test_source, y_pred_source)
    # f1_source = f1_score(y_test_source, y_pred_source)
    # auc_source = roc_auc_score(y_test_source, y_prob_source)

    # # Step 4: Print the metrics
    # print(f"Accuracy: {accuracy_source:.4f}")
    # print(f"Precision: {precision_source:.4f}")
    # print(f"Recall: {recall_source:.4f}")
    # print(f"F1-Score: {f1_source:.4f}")
    # print(f"AUC: {auc_source:.4f}")

    # # Display the classification report (Precision, Recall, F1-score)
    # print("\nClassification Report:\n", classification_report(y_test_source, y_pred_source))

    # # Display the confusion matrix
    # print("\nConfusion Matrix:\n", confusion_matrix(y_test_source, y_pred_source))
    #Load the saved black-box model
    rf_model = joblib.load('Path/Trained_BB_models/gsvm_model_source.pkl')
    # Make predictions on the test set
    # y_pred_source = rf_model.predict(X_test_source)
    # y_prob_source = rf_model.predict_proba(X_test_source)[:, 1]

    # Evaluate the model's performance
    # accuracy_source = accuracy_score(y_test_source, y_pred_source)
    # precision_source = precision_score(y_test_source, y_pred_source)
    # recall_source = recall_score(y_test_source, y_pred_source)
    # f1_source = f1_score(y_test_source, y_pred_source)
    # auc_source = roc_auc_score(y_test_source, y_prob_source)

    # Print the metrics
    # print(f"Accuracy: {accuracy_source:.4f}")
    # print(f"Precision: {precision_source:.4f}")
    # print(f"Recall: {recall_source:.4f}")
    # print(f"F1-Score: {f1_source:.4f}")
    # print(f"AUC: {auc_source:.4f}")

    # # Display the classification report (Precision, Recall, F1-score)
    # print("\nClassification Report:\n", classification_report(y_test_source, y_pred_source))
    return X, y, X_train_source, X_test_source, y_train_source, y_test_source, rf_model
