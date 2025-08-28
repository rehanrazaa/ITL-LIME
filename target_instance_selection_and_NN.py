import pandas as pd
import numpy as np
from target_domain import get_target_data

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

def get_target_instance():

    X_target, y_target, X_train_target, X_target_test, y_train_target, y_target_test, rf_model_target = get_target_data()


    # Select a single instance from the target training set (e.g., the 20th instance)
    target_instance_index = 15 # You can change this index to select a different instance
    target_instance = X_target_test.iloc[target_instance_index]
    # Reshape the instance to match the model's expected input shape (1 sample, n features) and keep feature names
    target_instance_reshaped = pd.DataFrame([target_instance], columns=X_target_test.columns)
    # Predict the class for the selected instance
    predicted_class = rf_model_target.predict(target_instance_reshaped)
    # Predict the probabilities for each class
    prediction_probabilities = rf_model_target.predict_proba(target_instance_reshaped)
    # Get the original label for the selected instance from the test set
    original_label = y_target_test.iloc[target_instance_index]
    print("------------------------------------------------------------------------------------------")
    # Print the details
    print("Selected Target Instance Features:")
    print(target_instance)
    print("------------------------------------------------------------------------------------------")
    print("\nOriginal Label:")
    print(original_label)

    print("\nModel Prediction:")
    print(predicted_class[0])

    print("\nPrediction Probabilities (for each class):")
    print(prediction_probabilities)

    # Use argmax to get the class with the highest probability
    predicted_class = np.argmax(prediction_probabilities)
    print(f"Predicted Class: {predicted_class}")

    return target_instance_index, target_instance

# Function to select nearest instances within a distance threshold
def select_nearest_instances_with_threshold(X_target, target_instance, num_neighbors=200, distance_threshold=2.0):
    target_instance_df = pd.DataFrame([target_instance], columns=X_target.columns)

    # Create a KNN model and fit to the entire dataset
    knn = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean')
    knn.fit(X_target)  # Fit to the entire target dataset

    # Find the nearest neighbors for the target instance
    distances, indices = knn.kneighbors(target_instance_df)  # Get indices and distances of the nearest neighbors

    # Apply the distance threshold: only keep neighbors within the specified distance threshold
    valid_indices = indices[0][distances[0] <= distance_threshold]  # Correct the indexing

    # If the number of valid neighbors is less than the desired size, adjust the size
    if len(valid_indices) < num_neighbors:
        print(f"Warning: Only {len(valid_indices)} neighbors were found within the distance threshold.")

    # Get the corresponding instances
    nearest_instances = X_target.iloc[valid_indices]

    return nearest_instances, len(valid_indices)


def get_target_nn_instances():

    _, target_instance = get_target_instance()
    X_target,_,_,_,_,_,_ = get_target_data()

    nearest_instances, num_selected = select_nearest_instances_with_threshold(X_target, target_instance, num_neighbors=300, distance_threshold=15.0)

    # Display the number of instances selected
    print(f"Number of selected instances: {num_selected}")
    target_instances_lime = nearest_instances

    return target_instances_lime

#
# get_target_nn_instances()
