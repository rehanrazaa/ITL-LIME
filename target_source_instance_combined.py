from source_prototype_clusters_selection import get_source_cluster_instances
from target_instance_selection_and_NN import get_target_nn_instances
from sklearn.model_selection import train_test_split
import pandas as pd

def get_combined_instances():
    target_instances_lime = get_target_nn_instances()
    source_instances_lime = get_source_cluster_instances()
    # Combine Source & Target Perturbations
    X_combined_instances = pd.concat([target_instances_lime, source_instances_lime], axis=0).reset_index(drop=True)
    #Split into Train, Validation, and Test for SCARF (After Combining source+target real isntance) to learn representation
    X_instance_train, X_instance_valid = train_test_split(X_combined_instances, test_size=0.2, random_state=42)
    print(f"Combined instances returned")
    return X_combined_instances, X_instance_train, X_instance_valid
