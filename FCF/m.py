import json
import numpy as np
import pandas as pd
from FairClustering import FairClustering

def preprocess_data(df, dataset_name, sensitive_attribute, columns_to_drop, k_choice, k_value):
    print(f"Original shape: {df.shape}")
    if df.columns[0] != sensitive_attribute:
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index(sensitive_attribute)))
        df = df[cols]

    if sensitive_attribute in df.columns:
        colors = df[sensitive_attribute].to_numpy()
        df.drop(sensitive_attribute, axis=1, inplace=True)
        print(f"Unique values in {sensitive_attribute} after correction:", np.unique(colors, return_counts=True))
    else:
        colors = None

    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    df.fillna(df.mean(), inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    print("Final DataFrame info:")
    df.info()
    dataset = df.to_numpy(dtype=float)
    
    # Determine the number of clusters K
    if k_choice == 'algorithm':
        # K = log_means(dataset, 5, 10)  # Example range [2, 10]
        print(f"Algorithm determined K: {K}")
    elif k_choice == 'user':
        K = k_value
        print(f"User specified K: {K}")
    else:
        K = 5  # Default value
        print(f"Default K: {K}")

    # Debugging print to confirm the structure and first few rows of the dataset
    print("First few rows of the dataset after preprocessing:\n", dataset[:5, :])

    return dataset, colors, K

def load_and_preprocess_data(dataset_path, dataset_name, k_choice, k_value):
    df = pd.read_csv(dataset_path)
    if df is not None:
        if dataset_name == "bank":
            return preprocess_data(df, dataset_name, "marital", ['job', 'age', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome'], k_choice, k_value)
        elif dataset_name == "adult":
            return preprocess_data(df, dataset_name, "sex", ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country', 'income'], k_choice, k_value)
        elif dataset_name == "diabetes":
            age_buckets = {'[70-80)': 75, '[60-70)': 65, '[50-60)': 55, '[80-90)': 85, '[40-50)': 45, '[30-40)': 35, '[90-100)': 95, '[20-30)': 25, '[10-20)': 15, '[0-10)': 5}
            df['age'] = df['age'].map(age_buckets)
            return preprocess_data(df, dataset_name, "gender", ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'], k_choice, k_value)
        else:
            raise ValueError("Unsupported dataset. Please choose 'adult', 'bank', or 'diabetes' or edit the code to add another dataset.")
    else:
        print("Failed to load data.")
        return None, None, None

def split_data(data, colors, split_size, random_state):
    np.random.seed(random_state)
    
    # Split data into majority and minority classes based on colors
    majority_class_indices = np.where(colors == 1)[0]
    minority_class_indices = np.where(colors == 0)[0]
    
    # Debugging print to confirm the split indices
    print(f"Majority class indices: {majority_class_indices[:5]}")
    print(f"Minority class indices: {minority_class_indices[:5]}")
    
    # Sample from the majority and minority classes
    majority_sample_indices = np.random.choice(majority_class_indices, size=max(split_size), replace=False)
    minority_sample_indices = np.random.choice(minority_class_indices, size=min(split_size), replace=False)
    
    # Combine the sampled indices
    combined_indices = np.concatenate((majority_sample_indices, minority_sample_indices))
    
    # Shuffle the combined indices
    np.random.shuffle(combined_indices)
    
    # Get the combined data and colors based on the shuffled indices
    combined_data = data[combined_indices]
    combined_colors = colors[combined_indices]
    
    # Identify blues and reds based on the combined_colors
    blues = [i for i, color in enumerate(combined_colors) if color == 1]
    reds = [i for i, color in enumerate(combined_colors) if color == 0]
    
    # Debugging print to confirm the combined data and colors
    print("Combined data first few rows after split:\n", combined_data[:5])
    print("Combined colors first few rows after split:\n", combined_colors[:5])
    
    return combined_data.tolist(), blues, reds


def main():
    dataset_path = '/Users/ankitasaha/Downloads/Fair-Clustering Codes/Fair Clustering Final/data/Adult/adult2.csv'
    dataset_name = 'adult'
    k_choice = 'user'
    k_value = 5
    config_path = 'config.json'
    degrees = 10

    data, colors, _ = load_and_preprocess_data(dataset_path, dataset_name, k_choice, k_value)
    print(f"First few rows of data:\n{data[:5]}")
    print(f"Colors:\n{colors[:5]}")
    if data is None or colors is None:
        print("Data loading and preprocessing failed.")
        return

    # Ensuring split size is balanced according to the smallest class size
    split_size = (min(np.sum(colors == 1), np.sum(colors == 0)), min(np.sum(colors == 1), np.sum(colors == 0)))
    combined_data, blues, reds = split_data(data, colors, split_size, random_state=42)

    # Verify that combined_data is correctly aligned with colors
    print(f"First few rows of combined data:\n{combined_data[:5]}")
    print(f"Number of blues: {len(blues)}")
    print(f"Number of reds: {len(reds)}")

    fair_clustering = FairClustering(config_path, combined_data, blues, reds, colors)
    fair_clustering.fit(degrees, decomposition_type="vanilla")
    results = fair_clustering.evaluate()
    print("Results:", results)

if __name__ == "__main__":
    main()
