import numpy as np
def calculate_kmedian_cost(centroids, dataset):
    
    "Computes and returns k-median cost for given dataset and centroids"
    return sum(np.amin(np.concatenate([np.linalg.norm(dataset - centroid, axis=1).reshape((dataset.shape[0], 1)) for centroid in centroids], axis=1), axis=1))


def fair_kmedian_cost(centroids, dataset, fairlet_decomposition):
    total_cost = 0
    for i in range(len(fairlet_decomposition.fairlets)):
        cost_list = [np.linalg.norm(dataset[centroids[j], :] - dataset[fairlet_decomposition.fairlet_centers[i], :]) for j in range(len(centroids))]
        cost, j = min((cost, j) for (j, cost) in enumerate(cost_list))
        total_cost += sum([np.linalg.norm(dataset[centroids[j], :] - dataset[point, :]) for point in fairlet_decomposition.fairlets[i]])
    return total_cost
 