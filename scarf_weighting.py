import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from gower import gower_matrix
from sklearn.linear_model import LogisticRegression
from SCARF_training import get_scarf_encoder
from target_source_instance_combined import get_combined_instances
from target_instance_selection_and_NN import get_target_instance
import pandas as pd

def scarf_weighting():
    encoder = get_scarf_encoder()
    _, target_instance = get_target_instance()
    X_combined_instances, X_instance_train, X_instance_valid = get_combined_instances()
    # Encode in latent space 
    encoder.eval()
    with torch.no_grad():
        # Encode target instance
        encoded_x = encoder(torch.tensor(target_instance, dtype=torch.float32).unsqueeze(0)).numpy()  
        # Encode all combined perturbed samples
        encoded_perturbed = encoder(torch.tensor(X_combined_instances.values, dtype=torch.float32)).numpy()  

    # compute distance
    distances = gower_matrix(encoded_perturbed, encoded_x).flatten() 

    # assign weights
    kernel_width = 1.5 * np.sqrt(X_combined_instances.shape[1])  
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))  

    final_weights = weights 

    # Get closest samples based on highest weights
    topk_indices = np.argsort(final_weights)[-1000:]

    # Sort them by weight descending (most similar first)
    topk_sorted_indices = topk_indices[np.argsort(final_weights[topk_indices])[::-1]]

    # Retrieve selected samples and weights
    top_per_samples = X_combined_instances.iloc[topk_sorted_indices].reset_index(drop=True)
    top_per_weights = final_weights[topk_sorted_indices]

    # After selecting top samples and their final weights
    top_per_weights = top_per_weights / np.max(top_per_weights)

    return top_per_samples , top_per_weights
