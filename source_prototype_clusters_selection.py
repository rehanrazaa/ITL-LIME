import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from source_domain import get_source_data
from target_instance_selection_and_NN import get_target_instance



def get_source_cluster_instances():


    X, y, X_train_source, X_test_source, y_train_source, y_test_source, rf_model = get_source_data()
    target_instance_index, _ = get_target_instance()


    # Number of clusters (prototypes) to select
    n_prototypes = 15

    # Apply K-means clustering on the entire dataset
    kmeans = KMeans(n_clusters=n_prototypes, random_state=42, n_init=10)
    kmeans.fit(X)

    # For each cluster, find the closest data point to the cluster center to use as a prototype
    prototype_indices = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(X.values - center, axis=1)
        closest_index = np.argmin(distances)
        prototype_indices.append(X.index[closest_index])  # Get the original index

    # Extract prototypes based on the selected indices
    prototypes = X.loc[prototype_indices].copy()

    # Assuming `target_instance_index` is the row (instance) you want to find the best prototype for
    target_instance = X.iloc[target_instance_index]  # Set `target_instance_index` to the row index of your target

    # Calculate the distance between the target instance and each prototype
    distances_to_prototypes = np.linalg.norm(prototypes.values - target_instance.values, axis=1)

    # Find the closest prototype (the one with the minimum distance)
    best_prototype_index = np.argmin(distances_to_prototypes)
    best_prototype = prototypes.iloc[best_prototype_index]

    # Print the best prototype (closest to the target instance)
    print("------------------------------------------------------------------------------------------")
    print("\nBest Prototype (Closest to Target Instance):")
    print(best_prototype)
    print("------------------------------------------------------------------------------------------")

    # Now, we will check the data assigned to each prototype,
    # and print the total number of data points and label counts (+ve, -ve) under each prototype
    print("\nPrototype Data Counts and Class Label Breakdown:")

    # For each cluster (prototype), get the points assigned to it and count their class labels
    cluster_assignments = kmeans.predict(X)

    # Initialize variables to track the best prototype data counts
    best_prototype_count = 0
    best_prototype_positive_count = 0
    best_prototype_negative_count = 0

    # For each prototype (cluster), calculate the number of points in the cluster
    for i in range(n_prototypes):
        # Get indices of data points assigned to the current cluster
        cluster_indices = np.where(cluster_assignments == i)[0]
        cluster_data = X.iloc[cluster_indices]
        cluster_labels = y.iloc[cluster_indices]

        # Count the class labels (+ve and -ve)
        positive_count = (cluster_labels == 1).sum()  # Assuming +ve is labeled as 1
        negative_count = (cluster_labels == 0).sum()  # Assuming -ve is labeled as 0

        # Print the results for the current cluster (prototype)
        # print(f"\nPrototype {i + 1}:")
        # print(f"  Total Data Points: {len(cluster_data)}")
        # print(f"  Positive (+ve) Class Count: {positive_count}")
        # print(f"  Negative (-ve) Class Count: {negative_count}")

        # Track the best prototype (the one closest to the target instance)
        if i == best_prototype_index:
            best_prototype_count = len(cluster_data)
            best_prototype_positive_count = positive_count
            best_prototype_negative_count = negative_count

    # Print the details of the best prototype
    print("------------------------------------------------------------------------------------------")
    print(f"\nBest Prototype Details (Closest to Target Instance):")
    print(f"  Prototype Index: {best_prototype_index + 1}")  # Displaying 1-based index
    print(f"  Total Data Points: {best_prototype_count}")
    print(f"  Positive (+ve) Class Count: {best_prototype_positive_count}")
    print(f"  Negative (-ve) Class Count: {best_prototype_negative_count}")
    print("------------------------------------------------------------------------------------------")

    # Step 1: Get the cluster assignment for each instance in the dataset
    cluster_assignments = kmeans.predict(X)

    # Step 2: Find all instances assigned to the best prototype (the closest prototype to the target instance)
    best_prototype_cluster = best_prototype_index  # The index of the best prototype

    # Step 3: Extract the indices of all instances assigned to the best prototype
    best_prototype_indices = np.where(cluster_assignments == best_prototype_cluster)[0]

    # Step 4: Store the real instances assigned to the best prototype in the `data` variable
    data = X.iloc[best_prototype_indices]

    # Optionally, if you want to include the corresponding class labels (`y`), you can also:
    data_with_labels = pd.concat([data, y.iloc[best_prototype_indices]], axis=1)
    data_with_labels.shape
    data_with_labels = data_with_labels.drop(columns=['Depression'])
    source_instances_lime = data_with_labels
    # Sample 500 instances
    source_instances_lime = source_instances_lime.sample(n=450, random_state=42)

    print("Done")


    return source_instances_lime



#get_source_cluster_instances()




