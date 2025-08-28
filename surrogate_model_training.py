from black_box_prediction import get_black_box_prediction
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
from sklearn.svm import SVC
from scarf_weighting import scarf_weighting

y_blackbox_target_pred_class = get_black_box_prediction()
top_per_samples , top_per_weights = scarf_weighting()

# Initialize StratifiedKFold (5-fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store fidelity metrics for each fold
accuracy_train_scores = []
f1_train_scores = []
precision_train_scores = []
recall_train_scores = []
auc_train_scores = []

accuracy_test_scores = []
f1_test_scores = []
precision_test_scores = []
recall_test_scores = []
auc_test_scores = []

# Loop through the splits
for train_index, test_index in skf.split(top_per_samples, y_blackbox_target_pred_class):
    # Use iloc to properly index DataFrames/Series with integer indices
    X_train, X_test = top_per_samples.iloc[train_index], top_per_samples.iloc[test_index]
    y_train, y_test = y_blackbox_target_pred_class[train_index], y_blackbox_target_pred_class[test_index]
    weights_train, weights_test = top_per_weights[train_index], top_per_weights[test_index]

    # Train the Linear SVM model as an interpretable (surrogate) model
    svm_model_target = SVC(kernel='linear', C=1000.0, class_weight='balanced')
    svm_model_target.fit(X_train, y_train, sample_weight=weights_train)

    # Predict on training set
    y_train_pred_class = svm_model_target.predict(X_train)
    y_train_pred_binary = (y_train_pred_class > 0).astype(int)
    y_train_binary = (y_train > 0).astype(int)

    # Predict on test set
    y_test_pred_class = svm_model_target.predict(X_test)
    y_test_pred_binary = (y_test_pred_class > 0).astype(int)
    y_test_binary = (y_test > 0).astype(int)

    # Calculate fidelity for the training set
    accuracy_train = accuracy_score(y_train_binary, y_train_pred_binary)
    f1_train = f1_score(y_train_binary, y_train_pred_binary, zero_division='warn')
    precision_train = precision_score(y_train_binary, y_train_pred_binary, zero_division='warn')
    recall_train = recall_score(y_train_binary, y_train_pred_binary, zero_division='warn')
    if len(np.unique(y_train_binary)) == 2:
        auc_train = roc_auc_score(y_train_binary, y_train_pred_binary)
    else:
        auc_train = 'Undefined (only one class present)'

    # Store training fidelity metrics
    accuracy_train_scores.append(accuracy_train)
    f1_train_scores.append(f1_train)
    precision_train_scores.append(precision_train)
    recall_train_scores.append(recall_train)
    auc_train_scores.append(auc_train)

    # Calculate fidelity for the test set
    accuracy_test = accuracy_score(y_test_binary, y_test_pred_binary)
    f1_test = f1_score(y_test_binary, y_test_pred_binary, zero_division='warn')
    precision_test = precision_score(y_test_binary, y_test_pred_binary, zero_division='warn')
    recall_test = recall_score(y_test_binary, y_test_pred_binary, zero_division='warn')
    if len(np.unique(y_test_binary)) == 2:
        auc_test = roc_auc_score(y_test_binary, y_test_pred_binary)
    else:
        auc_test = 'Undefined (only one class present)'

    # Store test fidelity metrics
    accuracy_test_scores.append(accuracy_test)
    f1_test_scores.append(f1_test)
    precision_test_scores.append(precision_test)
    recall_test_scores.append(recall_test)
    auc_test_scores.append(auc_test)

# Compute mean and std for all metrics (training and test)
def compute_mean_std(metrics_scores):
    mean_score = np.mean(metrics_scores)
    std_score = np.std(metrics_scores)
    return mean_score, std_score

# Calculate mean and std for training and test metrics
accuracy_train_mean, accuracy_train_std = compute_mean_std(accuracy_train_scores)
f1_train_mean, f1_train_std = compute_mean_std(f1_train_scores)
precision_train_mean, precision_train_std = compute_mean_std(precision_train_scores)
recall_train_mean, recall_train_std = compute_mean_std(recall_train_scores)
auc_train_mean, auc_train_std = compute_mean_std(auc_train_scores)

accuracy_test_mean, accuracy_test_std = compute_mean_std(accuracy_test_scores)
f1_test_mean, f1_test_std = compute_mean_std(f1_test_scores)
precision_test_mean, precision_test_std = compute_mean_std(precision_test_scores)
recall_test_mean, recall_test_std = compute_mean_std(recall_test_scores)
auc_test_mean, auc_test_std = compute_mean_std(auc_test_scores)

# Print the results
print("------------------------------------------------------------------------------------------")
print(f"Training Fidelity Accuracy: {accuracy_train_mean:.4f} ± {accuracy_train_std:.4f}")
print(f"Training Fidelity F1-Score: {f1_train_mean:.4f} ± {f1_train_std:.4f}")
print(f"Training Fidelity Precision: {precision_train_mean:.4f} ± {precision_train_std:.4f}")
print(f"Training Fidelity Recall: {recall_train_mean:.4f} ± {recall_train_std:.4f}")
print(f"Training Fidelity AUC: {auc_train_mean:.4f} ± {auc_train_std:.4f}")
print("------------------------------------------------------------------------------------------")
print(f"Test Fidelity Accuracy: {accuracy_test_mean:.4f} ± {accuracy_test_std:.4f}")
print(f"Test Fidelity F1-Score: {f1_test_mean:.4f} ± {f1_test_std:.4f}")
print(f"Test Fidelity Precision: {precision_test_mean:.4f} ± {precision_test_std:.4f}")
print(f"Test Fidelity Recall: {recall_test_mean:.4f} ± {recall_test_std:.4f}")
print(f"Test Fidelity AUC: {auc_test_mean:.4f} ± {auc_test_std:.4f}")
print("------------------------------------------------------------------------------------------")
