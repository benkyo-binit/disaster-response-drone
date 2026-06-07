import pandas as pd
from sklearn.model_selection import train_test_split
from config import CSV_PATH, TRAIN_RATIO, VAL_RATIO, SEED

def split_dataset():
    df = pd.read_csv(CSV_PATH)
    train_df, temp_df = train_test_split(df, test_size=1 - TRAIN_RATIO, stratify=df['label'], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=VAL_RATIO / (1 - TRAIN_RATIO), stratify=temp_df['label'], random_state=SEED)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
