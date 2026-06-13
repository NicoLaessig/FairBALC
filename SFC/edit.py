import pandas as pd

# Assuming your dataset is stored in a CSV file named 'bank_dataset.csv'
file_path = "bank.csv"

# Read the dataset into a pandas DataFrame
df = pd.read_csv(file_path, delimiter=";")

# Shift the "marital" column to the first position
marital_col = df.pop('marital')
df.insert(0, 'marital', marital_col)

# Mapping to change values in the "marital" column if needed


# Save the modified DataFrame as a new CSV file
output_file = "bank2.csv"
df.to_csv(output_file, index=False)

print(f"Modified dataset saved as '{output_file}'.")
