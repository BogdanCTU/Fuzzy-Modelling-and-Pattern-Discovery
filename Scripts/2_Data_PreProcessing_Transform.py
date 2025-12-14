import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import os
import sqlite3

warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV using European decimal format."""
    print(f"--- Loading Data from {filepath} ---")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file '{filepath}' was not found.")
    
    df = pd.read_csv(filepath, sep=';', decimal=',', low_memory=False)
    return df

from sklearn.preprocessing import MinMaxScaler

def create_transformed_dataset_positive(df: pd.DataFrame, feature_range=(0,1)) -> pd.DataFrame:
    """Transform features to a strictly positive range using Min-Max scaling."""
    
    # Filter valid bodyweights
    df_valid = df[df['BodyweightKg'] > 0].copy().reset_index(drop=True)
    print(f"Original rows: {len(df)}, Valid rows: {len(df_valid)}")

    # Feature construction
    df_valid['Strength_to_Weight_Ratio'] = df_valid['TotalKg'] / df_valid['BodyweightKg']

    # Columns to scale
    cols_to_scale = ['BodyweightKg', 'TotalKg', 'Strength_to_Weight_Ratio']
    optional_cols = ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'Age']
    cols_to_scale += [c for c in optional_cols if c in df_valid.columns]

    # Apply Min-Max scaling to [feature_range[0], feature_range[1]]
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_values = scaler.fit_transform(df_valid[cols_to_scale])
    df_scaled = pd.DataFrame(scaled_values, columns=[f"{c}_Scaled" for c in cols_to_scale])

    # Combine original + scaled features
    df_transformed = pd.concat([df_valid, df_scaled], axis=1)
    
    return df_transformed



def save_transformed_to_csv(df: pd.DataFrame, filename: str):
    """Save transformed dataset to CSV (European format)."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].round(6)
    df.to_csv(filename, index=False, sep=';', decimal=',', float_format='%.6f')
    print(f"Saved transformed dataset to CSV: {filename}")

def save_transformed_to_sqlite(df: pd.DataFrame, db_filename: str, table_name: str = 'transformed_data'):
    """Save transformed dataset to SQLite database."""
    conn = sqlite3.connect(db_filename)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Saved transformed dataset to SQLite database: {db_filename}, table: {table_name}")

def main(file_path: str):
    df_original = load_data(file_path)
    df_transformed = create_transformed_dataset_positive(df_original)

    # Save externally without touching original
    save_transformed_to_csv(df_transformed, 'phase2_transformed.csv')
    save_transformed_to_sqlite(df_transformed, 'phase2_transformed.db')

if __name__ == "__main__":
    main('Phase1_open_powerlifting_dataset_cleaned.csv')
