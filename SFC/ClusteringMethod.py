import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from SFC.FairletDecomposition import FairletDecomposition
from SFC.KMedianCost import calculate_kmedian_cost, fair_kmedian_cost
from SFC.QuadTree import build_quadtree, calculate_balance_score
from SFC.TreeNode import *
from sklearn.metrics import silhouette_score

class ScalableClustering:
    def __init__(self, p, q, k, dataset_path=None, sample_size=None):
        self.p = p
        self.q = q
        self.k = k
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.dataset = None
        self.colors = None
        self.fairlet_decomposition = FairletDecomposition()

    def load_data(self):
        if self.dataset_path:
            df = pd.read_csv(self.dataset_path)
            print("Original shape:", df.shape)
            return df
        else:
            print("Dataset path not provided.")
            return None
            
    def fit(self):
        # Build the quadtree
        root = build_quadtree(self.dataset)

        # Perform fairlet decomposition
        self.fairlet_decomposition.tree_fairlet_decomposition(self.p, self.q, root, self.dataset, self.colors)

        # Extract fairlet centers for k-medoids clustering
        fairlet_center_indices = self.fairlet_decomposition.fairlet_centers
        if len(fairlet_center_indices) == 0:
          raise ValueError("No fairlet centers identified. Please check the fairlet decomposition process.")
        
        fairlet_centers = self.dataset[fairlet_center_indices]
        print(f"Number of fairlet centers: {len(np.unique(fairlet_center_indices))}")
        
        # Ensure there are enough fairlet centers for the desired number of clusters
        if len(np.unique(fairlet_center_indices)) < self.k:
          print("Warning: Number of fairlet centers less than n_clusters. Adjusting n_clusters.")
          self.k = max(1, len(np.unique(fairlet_center_indices)) - 1)  # Ensure at least 1 cluster

        # Proceed with k-medoids clustering if there are enough centers
        if len(np.unique(fairlet_center_indices)) > 1:  # Ensure more than one center for clustering
           self.kmedoids = KMedoids(n_clusters=self.k, random_state=42).fit(fairlet_centers)
        else:
           raise ValueError("Insufficient fairlet centers for clustering.")
        
        if np.isnan(fairlet_centers).any():
           raise ValueError("Fairlet centers contain NaNs.")

        # Perform k-medoids clustering on fairlet centers
        self.kmedoids = KMedoids(n_clusters=self.k, random_state=42).fit(fairlet_centers)

    def predict(self, X):
        if hasattr(self, 'kmedoids'):
            return self.kmedoids.predict(X)
        else:
            raise RuntimeError("The model has not been fitted yet.")

    def evaluate(self, labels=None):
        if labels is None:
            # Calculate and print the k-median cost using the determined centroids
            centroids_indices = self.kmedoids.medoid_indices_
            centroids = self.dataset[centroids_indices]
            k_median_cost = calculate_kmedian_cost(centroids, self.dataset)
            print(f"k-Median Cost: {k_median_cost}")

            # Optional: Calculate balance score if relevant
            # This step requires labels for the entire dataset, which might be obtained differently
        
            labels = self.predict(self.dataset)  # Example of getting labels if needed
        balance_score = calculate_balance_score(labels, self.colors)
        print(f"Balance Score: {balance_score}")

        silhouette_avg = silhouette_score(self.dataset, labels)
        print(f"Silhouette Score: {silhouette_avg}")

        return balance_score, silhouette_avg

