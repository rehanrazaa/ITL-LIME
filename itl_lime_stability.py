import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import torch
from itertools import combinations
from source_prototype_clusters_selection import get_source_cluster_instances
from target_instance_selection_and_NN import get_target_nn_instances
from sklearn.linear_model import LogisticRegression
# ---------------------- CONFIG ----------------------
target_instances_lime = get_target_nn_instances()
source_instances_lime = get_source_cluster_instances()
from target_domain import get_target_data
from target_instance_selection_and_NN import get_target_instance
from SCARF_training import get_scarf_encoder
from scarf_weighting import scarf_weighting

NUM_RUNS = 5
TOP_K = 3
NUM_SAMPLES = 2000

# Replace these with your actual feature lists
categorical_features = ['gender_encoded', 'location_encoded', 'smoking_history_encoded', 'race_encoded', 'hypertension', 'heart_disease']
continuous_features = ['age_scaled', 'bmi_scaled', 'hbA1c_level_scaled', 'blood_glucose_level_scaled']
all_features = categorical_features + continuous_features

# ---------------------- HELPERS ----------------------
def encode_with_encoder(encoder, df):
    encoder.eval()
    with torch.no_grad():
        encoded = encoder(torch.tensor(df.values, dtype=torch.float32)).numpy()
    return encoded

def compute_weights(encoded_all, encoded_instance):
    distances = euclidean_distances(encoded_all, encoded_instance).flatten()
    kernel_width = 0.5 * np.sqrt(encoded_all.shape[1])
    return np.exp(-(distances ** 2) / (kernel_width ** 2))

def get_top_k_features(model, feature_names, k):
    coefs = np.abs(model.coef_[0])
    top_indices = np.argsort(coefs)[-k:]
    return set([feature_names[i] for i in top_indices])

def compute_jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

# ---------------------- MAIN STABILITY COMPUTATION ----------------------
def compute_lime_stability(target_instance, encoder, predict_proba_fn):
    top_feature_sets = []

    for run in range(NUM_RUNS):
        # Set target and source real instances
        perturbed_target = target_instances_lime
        perturbed_source = source_instances_lime

        # Scaling continuous
        #scaler = StandardScaler()
        combined = pd.concat([perturbed_target, perturbed_source]).reset_index(drop=True)
        #  Encode all samples in Encoder latent space
        encoded_all = encode_with_encoder(encoder, combined)
        encoded_instance = encode_with_encoder(encoder, target_instance.to_frame().T)
        # Create domain labels
        n_target_samples = perturbed_target.shape[0]
        n_source_samples = perturbed_source.shape[0]
        labels_domain = np.concatenate([
            np.zeros(n_target_samples),  # target samples label 0
            np.ones(n_source_samples)    # source samples label 1
        ])
        # Compute locality weights
        distances = euclidean_distances(encoded_all, encoded_instance).flatten()
        kernel_width = 0.5 * np.sqrt(encoded_all.shape[1])
        locality_weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        final_weights = locality_weights
        # Normalize final weights (important for stability)
        final_weights = final_weights / np.max(final_weights)
        # Select top samples based on final weights
        top_indices = np.argsort(final_weights)[-NUM_SAMPLES:]
        top_sorted = top_indices[np.argsort(final_weights[top_indices])[::-1]]
        top_samples = combined.iloc[top_sorted].reset_index(drop=True)
        top_weights = final_weights[top_sorted]
        # Get black-box predictions and labels
        probs = predict_proba_fn(top_samples)
        labels = 2 * (probs[:, 1] > 0.5).astype(int) - 1  # {-1, +1}
        # Fit surrogate model (Linear SVM)
        surrogate = LinearSVC()
        surrogate.fit(top_samples, labels, sample_weight=top_weights)
        # Get top-k important features
        top_features = get_top_k_features(surrogate, combined.columns.tolist(), TOP_K)
        top_feature_sets.append(top_features)
    # Compute Jaccard similarities
    similarities = []
    for a, b in combinations(top_feature_sets, 2):
        sim = compute_jaccard(a, b)
        similarities.append(sim)

    avg_similarity = np.mean(similarities)
    avg_distance = 1 - avg_similarity
    return avg_similarity, avg_distance, top_feature_sets

X_target, _, _, _, _, _, rf_model_target = get_target_data()
_,target_instance  = get_target_instance()
encoder = get_scarf_encoder()
top_per_samples , top_per_weights = scarf_weighting()

def predict_target_proba(input_data_target):
    input_df_target = pd.DataFrame(input_data_target, columns=X_target.columns)
    return rf_model_target.predict_proba(input_df_target)
    
avg_sim, avg_dist, all_top_features = compute_lime_stability(
    target_instance, encoder, predict_target_proba
)
print("------------------------------------------------------------------------------------------")
print("Average Jaccard Similarity (Stability):", avg_sim)
print("Top Features from Each Run:", all_top_features)
print("------------------------------------------------------------------------------------------")
