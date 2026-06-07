import pandas as pd

df = pd.read_csv('dataset_cleaned.csv')
print(df.head())
print("Unique labels:", df['label'].unique())