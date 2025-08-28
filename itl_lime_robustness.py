from sklearn.svm import SVC
from sklearn.metrics.pairwise import euclidean_distances
import torch
import numpy as np
import pandas as pd
from target_domain import get_target_data
from target_instance_selection_and_NN import get_target_instance
from SCARF_training import get_scarf_encoder
from scarf_weighting import scarf_weighting

# ---------------------- HELPER ----------------------
X_target, _, _, _, _, _, rf_model_target = get_target_data()
_,target_instance  = get_target_instance()
encoder = get_scarf_encoder()
top_per_samples , top_per_weights = scarf_weighting()

def encode_instance(encoder, df):
    """
    Encodes a single instance or batch of instances using the provided encoder.
    Handles BatchNorm layers safely for single instances.
    """
    encoder.eval() 
    tensor_input = torch.tensor(df.values, dtype=torch.float32)
    if tensor_input.ndim == 1:
        tensor_input = tensor_input.unsqueeze(0)  
    with torch.no_grad():
        encoded = encoder(tensor_input).numpy()
    return encoded

# ---------------------- LLE COMPUTATION ----------------------
def compute_lle(target_instance, neighbor_instances, encoder, rf_model_target, X_target, kernel_width=1.5):
    """
    Compute the Local Lipschitz Estimate (LLE) for a given instance and its neighborhood.
    """
    # Encode target instance
    encoded_target = encode_instance(encoder, target_instance[X_target.columns])

    # Encode neighbors
    encoded_neighbors = encode_instance(encoder, neighbor_instances[X_target.columns])

    # Compute RBF weights based on encoder distance
    distances = euclidean_distances(encoded_neighbors, encoded_target).flatten()
    kernel_width_scaled = kernel_width * np.sqrt(neighbor_instances.shape[1])
    weights = np.exp(-(distances ** 2) / (kernel_width_scaled ** 2))

    # Black-box predictions for neighbors
    def predict_fn(X):
        df = pd.DataFrame(X, columns=X_target.columns)
        return rf_model_target.predict_proba(df)

    y_neighbors_proba = predict_fn(neighbor_instances[X_target.columns].values)
    y_neighbors_class = 2 * (y_neighbors_proba[:, 1] > 0.5).astype(int) - 1  # {-1, +1}

    # Train surrogate model centered at target instance
    surrogate = SVC(kernel='linear', C=1.0, class_weight='balanced')
    surrogate.fit(neighbor_instances[X_target.columns].values, y_neighbors_class, sample_weight=weights)
    f_x = surrogate.coef_.flatten()

    # Compute LLE by comparing surrogate gradients for neighbors
    lle_vals = []

    for i, x_prime in neighbor_instances.iterrows():
        encoded_x_prime = encode_instance(encoder, x_prime[X_target.columns])
        dist_from_x_prime = euclidean_distances(encoded_neighbors, encoded_x_prime).flatten()
        weights_prime = np.exp(-(dist_from_x_prime ** 2) / (kernel_width_scaled ** 2))

        surrogate_prime = SVC(kernel='linear', C=10.0, class_weight='balanced')
        surrogate_prime.fit(neighbor_instances[X_target.columns].values, y_neighbors_class, sample_weight=weights_prime)
        f_x_prime = surrogate_prime.coef_.flatten()

        numerator = np.linalg.norm(f_x - f_x_prime)
        denominator = np.linalg.norm(target_instance[X_target.columns].values - x_prime[X_target.columns].values)

        if denominator > 1e-8:
            lle_vals.append(numerator / denominator)

    return max(lle_vals) if lle_vals else 0.0

# ---------------------- COMPUTE AND PRINT ----------------------
lle_value = compute_lle(target_instance, top_per_samples, encoder, rf_model_target, X_target)
print("------------------------------------------------------------------------------------------")
print(f"Average ITL-LIME LLE (Robustness): {lle_value:.4f}")
print("------------------------------------------------------------------------------------------")
