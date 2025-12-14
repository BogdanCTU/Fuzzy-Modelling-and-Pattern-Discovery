import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import os

# Suppress specific RuntimeWarnings for empty slices (handled in logic)
warnings.filterwarnings("ignore", category=RuntimeWarning) 

Loads the dataset.
- Uses sep=',' as specified.
- Sets low_memory=False to handle mixed types in columns like 'Tested' or 'State'.
def load_data(filepath):
    print(f"--- Loading Data from {filepath} ---")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file '{filepath}' was not found.") 
    return pd.read_csv(filepath, sep=',', low_memory=False)


### Helper function to save CSV in European format:
### - Separator: Semicolon (;)
### - Decimal: Comma (,)
def save_euro_format(df, filename):
    df.to_csv(filename, index=False, sep=';', decimal=',')
    print(f"Saved: {filename} (European Format)")


### Phase 1: Cleaning 
### - Reduces columns to target list.
### - Filters rows based on precise boundary conditions.
### - Saves: 'phase1_cleaned.csv'
def clean_data(df):
    print("--- Starting Cleaning Phase (Boundary Enforcement) ---")
    # 1. Column Reduction
    target_columns = [
        'Sex', 'Equipment', 'Age', 'AgeClass', 'BirthYearClass', 'BodyweightKg', 
        'WeightClassKg', 'Squat1Kg', 'Squat2Kg', 'Squat3Kg', 'Best3SquatKg', 
        'Bench1Kg', 'Bench2Kg', 'Bench3Kg', 'Best3BenchKg', 'Deadlift1Kg', 
        'Deadlift2Kg', 'Deadlift3Kg', 'Best3DeadliftKg', 'TotalKg', 
        'Dots', 'Tested' ]
    # Keep only columns that actually exist in the dataframe
    cols_to_keep = [col for col in target_columns if col in df.columns]
    # FIX: Add .copy() here to avoid SettingWithCopyWarning
    df = df[cols_to_keep].copy()
    print(f"Columns maintained: {len(df.columns)}")
    # Pre-step: Convert empty strings/whitespace to NaN
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    initial_row_count = len(df)
    # --- 2. Filtering based on Boundaries (Strict Exclusion) ---
    # 2.a Categorical Filters (Excluding empty or invalid)
    # Sex: Must be 'M' or 'F' (and not empty)
    if 'Sex' in df.columns:
        df = df[df['Sex'].isin(['M', 'F'])]
    # Equipment: Exclude empty
    if 'Equipment' in df.columns:
        df = df.dropna(subset=['Equipment'])
    # Sanctioned: Exclude empty
    if 'Sanctioned' in df.columns:
        df = df.dropna(subset=['Sanctioned'])
    # 2.b Tested Status: Only keep '1' (or 'Yes')
    if 'Tested' in df.columns:
        # Check against common positive indicators (1, 1.0, 'Yes')
        df = df[df['Tested'].astype(str).isin(['1', '1.0', 'Yes', 'yes'])]
        df['Tested'] = 1 # Standardize to 1     
    # 2.c Numeric Filters (Ensure numeric, exclude empty, check ranges [min, max))
    # Age [0, 120), BodyweightKg [0, 200), TotalKg [0, 1500)
    numeric_constraints = {
        'Age': (0, 120),
        'BodyweightKg': (0, 200),
        'TotalKg': (0, 1500) }
    # List of individual lift columns [0, 500)
    lift_columns = [
        'Squat1Kg', 'Squat2Kg', 'Squat3Kg', 'Best3SquatKg', 
        'Bench1Kg', 'Bench2Kg', 'Bench3Kg', 'Best3BenchKg', 
        'Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg', 'Best3DeadliftKg']
    for col in lift_columns:
        numeric_constraints[col] = (0, 500)
    # Apply all numeric constraints
    for col, (min_val, max_val) in numeric_constraints.items():
        if col in df.columns:
            # 1. Convert to numeric, coercing invalid entries (like strings) to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')        
            # 2. Exclude empty rows (NaNs)
            df = df.dropna(subset=[col])       
            # 3. Apply range filter [min, max) (min_val <= x < max_val)
            df = df[(df[col] >= min_val) & (df[col] < max_val)]
    final_row_count = len(df)
    print(f"Total rows dropped during strict cleaning: {initial_row_count - final_row_count}")
    # Output Phase 1
    save_euro_format(df, 'phase1_cleaned.csv')
    return df


### Ensure you point to the correct CSV file location
### The file path must match the location of your dataset
if __name__ == "__main__":
    main_preprocessing_pipeline('original_open_powerlifting_dataset.csv')
