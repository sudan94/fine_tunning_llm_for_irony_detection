import pandas as pd
from sklearn.model_selection import train_test_split

# Load your original dataset
original_data = pd.read_csv("merged.csv", encoding='utf-8')

# Split the dataset into train (80%) and temp (20%)
train_data, temp_data = train_test_split(original_data, test_size=0.2, random_state=42)

# Split the temp data into validation (10%) and test (10%)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save the train, validation, and test datasets to CSV files
train_data.to_csv("final/train.csv", encoding='utf-8', index=False)
validation_data.to_csv("final/validation.csv", encoding='utf-8', index=False)
test_data.to_csv("final/test.csv", encoding='utf-8', index=False)

print("Datasets split and saved successfully.")