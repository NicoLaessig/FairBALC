import numpy as np
import matplotlib.pyplot as plt

def distance(a, b, order=2):
	"""
	Calculates the specified norm between two vectors.
	
	Args:
		a (list) : First vector
		b (list) : Second vector
		order (int) : Order of the norm to be calculated as distance
	
	Returns:
		Resultant norm value
	"""
	assert len(a) == len(b), "Length of the vectors for distance don't match."
	return np.linalg.norm(x=np.array(a)-np.array(b), ord=order)

def balance_calculation(data, colors, centers, mapping):
    """
    Checks fairness for each of the clusters defined by k-centers.
    Returns balance using the total and class counts.
    
    Args:
        data (np.array)
        colors (np.array)
        centers (list)
        mapping (list) : tuples of the form (data_index, center_index)
        
    Returns:
        balance (float) : The minimum balance across all clusters
    """
    fair = {center: [0, 0] for center in centers}

    # Debugging the mapping and data
    print("Mapping first few rows:", mapping[:5])
    print("Data first few rows:", data[:5])
    print("Colors first few rows:", colors[:5])

    for i, center in mapping:
        sensitive_value = colors[i]  # Use the colors array for sensitive attribute
        if sensitive_value == 0:
            fair[center][0] += 1
        else:
            fair[center][1] += 1

    print(f"Fair dictionary: {fair}")

    balances = []
    for center in centers:
        p = fair[center][0]
        q = fair[center][1]
        print(f"Center: {center}, p: {p}, q: {q}")
        if p == 0 or q == 0:
            balance = 0
        else:
            balance = min(float(p) / q, float(q) / p)
        balances.append(balance)

    print(f"Balances: {balances}")

    return np.mean(balances)
def balance_calculation2(data, colors, centers, mapping):
    """
    Checks fairness for each of the clusters defined by k-means.
    Returns balance using the total and class counts.
    
    Args:
        data (np.array)
        colors (np.array)
        centers (list)
        mapping (list) : tuples of the form (data_index, center_index)
        
    Returns:
        balance (float) : The minimum balance across all clusters
    """
    # Convert centers to tuples
    centers = [tuple(center) for center in centers]
    
    fair = {center: [0, 0] for center in centers}

    for i, center_index in mapping:
        center = centers[center_index]
        sensitive_value = colors[i]  # Use the colors array for sensitive attribute
        if sensitive_value == 0:
            fair[center][0] += 1
        else:
            fair[center][1] += 1

    balances = []
    for center in centers:
        p = fair[center][0]
        q = fair[center][1]
        if p == 0 or q == 0:
            balance = 0
        else:
            balance = min(float(p) / q, float(q) / p)
        balances.append(balance)

    return np.mean(balances)
def balance_data(blues, reds):
    """
    Balances the number of blue and red points by randomly sampling without replacement.
    
    Args:
        blues (list): List of blue points indices.
        reds (list): List of red points indices.
        
    Returns:
        balanced_blues (list): Balanced list of blue points indices.
        balanced_reds (list): Balanced list of red points indices.
    """
    min_size = min(len(blues), len(reds))
    balanced_blues = np.random.choice(blues, min_size, replace=False).tolist()
    balanced_reds = np.random.choice(reds, min_size, replace=False).tolist()
    print(f"Balanced blues: {balanced_blues[:5]} (total: {len(balanced_blues)})")
    print(f"Balanced reds: {balanced_reds[:5]} (total: {len(balanced_reds)})")
    return balanced_blues, balanced_reds

def plot_analysis(degrees, costs, balances, step_size):
	"""
	Plots the curves for costs and balances.

	Args:
		degrees (list)
		costs (list)
		balances (list)
		step_size (int)
	"""
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
	ax[0].plot(costs, marker='.', color='blue')
	ax[0].set_xticks(list(range(0, len(degrees), step_size))) 
	ax[0].set_xticklabels(list(range(min(degrees), max(degrees)+1, step_size)), fontsize=12)
	ax[1].plot(balances, marker='x', color='saddlebrown')
	ax[1].set_xticks(list(range(0, len(degrees), step_size))) 
	ax[1].set_xticklabels(list(range(min(degrees), max(degrees)+1, step_size)), fontsize=12)
	plt.show()
	
