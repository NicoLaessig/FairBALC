from SFC.TreeNode import TreeNode
from collections import defaultdict
import numpy as np
import sys

EPSILON = 0.0001
def build_quadtree(dataset, max_levels=0, random_shift=True):
    if dataset.size == 0:
        print("Empty dataset provided to build_quadtree.")
        return None
    dimension = dataset.shape[1]
    try:
        lower = np.nanmin(dataset, axis=0)  # Using nanmin to ignore nan
        upper = np.nanmax(dataset, axis=0)  # Using nanmax to ignore nan
    except RuntimeWarning:
        print("Encountered an all-NaN axis. Check dataset preprocessing.")
        return None
    

    shift = np.zeros(dimension)
    if random_shift:
        for d in range(dimension):
            if np.isnan(lower[d]) or np.isnan(upper[d]):
                print(f"Warning: nan values detected in dimension {d}. Skipping this dimension.")
                continue  # Skip dimensions with nan values
            spread = upper[d] - lower[d]
            print(f"Dimension {d}: lower = {lower[d]}, upper = {upper[d]}, spread = {spread}")
            shift[d] = np.random.uniform(0, spread) if spread < sys.float_info.max else sys.float_info.max
            upper[d] += spread

    return build_quadtree_aux(dataset, range(dataset.shape[0]), lower, upper, max_levels, shift)

     
 
def build_quadtree_aux(dataset, cluster, lower, upper, max_levels, shift):
    """
    "lower" is the "bottom-left" (in all dimensions) corner of current hypercube
    "upper" is the "upper-right" (in all dimensions) corner of current hypercube
    """
 
    dimension = dataset.shape[1]
    cell_too_small = True
    for i in range(dimension):
        if upper[i]-lower[i] > EPSILON:
            cell_too_small = False
 
    node = TreeNode()
    if max_levels==1 or len(cluster)<=1 or cell_too_small:
        # Leaf
        node.set_cluster(cluster)
        return node
     
    # Non-leaf
    midpoint = 0.5 * (lower + upper)
    subclusters = defaultdict(list)
    for i in cluster:
        subclusters[tuple([dataset[i,d]+shift[d]<=midpoint[d] for d in range(dimension)])].append(i)
    for edge, subcluster in subclusters.items():
        sub_lower = np.zeros(dimension)
        sub_upper = np.zeros(dimension)
        for d in range(dimension):
            if edge[d]:
                sub_lower[d] = lower[d]
                sub_upper[d] = midpoint[d]
            else:
                sub_lower[d] = midpoint[d]
                sub_upper[d] = upper[d]
        node.add_child(build_quadtree_aux(dataset, subcluster, sub_lower, sub_upper, max_levels-1, shift))
    return node

def calculate_balance_score(labels, colors):
    unique_clusters = np.unique(labels)
    balance_scores = []

    for cluster in unique_clusters:
        indices = np.where(labels == cluster)[0]
        cluster_colors = np.array([colors[index] for index in indices])  # Convert to NumPy array

        count_red = np.sum(cluster_colors == 0)
        count_blue = np.sum(cluster_colors == 1)

        print(f"Cluster {cluster}: Red - {count_red}, Blue - {count_blue}")

        if count_red == 0 or count_blue == 0:
            balance = 0
        else:
            balance = min(count_red, count_blue) / max(count_red, count_blue)
        balance_scores.append(balance)

    return np.mean(balance_scores)