import pandas as pd

# Load your dataset
df = pd.read_csv('dataset.csv')

# Count the number of samples per class
label_counts = df['label'].value_counts()

# Filter out classes with less than 2 samples
valid_labels = label_counts[label_counts >= 2].index
df = df[df['label'].isin(valid_labels)]

# Optional: reset index after filtering
df = df.reset_index(drop=True)

# Save cleaned CSV (optional)
df.to_csv('dataset_cleaned.csv', index=False)

print(f"✅ Cleaned dataset. Remaining classes:\n{df['label'].value_counts()}")
