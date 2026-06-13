import time
import numpy as np
from sklearn.metrics import silhouette_score
from FCF.kcenters import KCenters
from FCF.fairlet_decomposition import VanillaFairletDecomposition, MCFFairletDecomposition
import json
from FCF.utils import balance_data, balance_calculation

class FairletClustering:
    def __init__(self, config, data, blues, reds, colors):
        self.config = config
        self.data = np.array(data)  # Ensure data is a numpy array
        self.blues = blues
        self.reds = reds
        self.colors = np.array(colors)

    def fit(self, degrees, decomposition_type="vanilla"):
        self.degrees = degrees

        # Balance the data before fairlet decomposition
        self.blues, self.reds = balance_data(self.blues, self.reds)

        # Fair K-means
        if decomposition_type == "vanilla":
            self.decomposition = VanillaFairletDecomposition(1, 2, self.blues, self.reds, self.data)
        elif decomposition_type == "mcf":
            self.decomposition = MCFFairletDecomposition(self.blues, self.reds, 2, self.config['distance_threshold'], self.data)
            self.decomposition.compute_distances()
            self.decomposition.build_graph(plot_graph=False)
        else:
            raise ValueError("Unsupported decomposition type. Please choose 'vanilla' or 'mcf'.")

        self.fairlets, self.fairlet_centers, self.fairlet_costs = self.decomposition.decompose()

        self.fair_results = {'degrees': [], 'costs': [], 'balances': [], 'silhouettes': []}
        for degree in range(3, degrees + 1, 1):
            start_time = time.time()
            kcenters = KCenters(k=degree)
            kcenters.fit(self.data)  # No fairlets argument here
            mapping = kcenters.assign()
            balance = balance_calculation(self.data, self.colors, kcenters.centers, mapping)
            labels = np.array([label for _, label in mapping])
            silhouette = silhouette_score(self.data, labels)

            self.fair_results['degrees'].append(degree)
            self.fair_results['costs'].append(kcenters.costs[-1])
            self.fair_results['balances'].append(balance)
            self.fair_results['silhouettes'].append(silhouette)

            print(f"Fair K-means - Degree {degree} - Time: {time.time() - start_time:.3f} seconds.")
            print(f"Fair results: Centers: {kcenters.centers}, Balance: {balance}, Silhouette: {silhouette}")

    def predict(self):
        time.sleep(0.1)  # Simulated delay
        avg_balance = np.mean(self.fair_results['balances'])
        avg_cost = np.mean(self.fair_results['costs'])
        best_balance = max(self.fair_results['balances'])
        best_cost = min(self.fair_results['costs'])
        return avg_balance, avg_cost

    def evaluate(self):
        avg_silhouette = np.mean(self.fair_results['silhouettes'])
        return avg_silhouette
