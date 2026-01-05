import pandas as pd
import numpy as np
import os

def load_data(path="data/raw/dataset.csv"):
    """Load the raw dataset and display initial info"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    
    df = pd.read_csv(path)
    print("Data loaded successfully")
    print("Initial data info:")
    print(df.info())
    print("Missing values per column:\n", df.isnull().sum())
    print(f"shape: {df.shape}")
    return df

def handle_missing_values(df):
    """Handle missing values and document decisions"""
    df_clean = df.copy()

    if df_clean.isnull().sum().sum() == 0:
        print("No missing values detected")
        return df_clean
    
    # filling categorical missing values with 'Unknown'
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            print(f"Filling {missing_count} missing values in '{col}' with 'Unknown'")
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Fill numerical missing values with median
    num_cols = df_clean.select_dtypes(include='number').columns
    for col in num_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            median_val = df_clean[col].median()
            print(f"Filling {missing_count} missing values in '{col}' with median: {median_val}")
            df_clean[col] = df_clean[col].fillna(median_val)
    
    return df_clean

def handle_outliers(df):
    """Remove outliers using vectorized IQR method for all numeric columns"""
    df_clean = df.copy()
    
    exclude_cols = ['id', 'popularity', 'duration_ms', 'key', 'mode', 
                    'time_signature', 'explicit']
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    num_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    initial_shape = df_clean.shape[0]

    Q1 = df_clean[num_cols].quantile(0.25)
    Q3 = df_clean[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = ((df_clean[num_cols] >= lower_bound) & (df_clean[num_cols] <= upper_bound)).all(axis=1)
    df_clean = df_clean[mask]

    removed_count = initial_shape - df_clean.shape[0]
    print(f"Removed {removed_count} outliers")
    print("Shape after outlier removal:", df_clean.shape)
    return df_clean


def convert_data_types(df):
    """Convert columns to appropriate data types"""
    df_clean = df.copy()

    # Boolean conversion
    if 'explicit' in df_clean.columns:
        df_clean['explicit'] = df_clean['explicit'].astype(bool)
    
    # Categorical conversion
    categorical_cols = ['artists', 'album_name', 'track_name', 'track_genre']
    float_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')

    for col in float_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('float32')        
    
    return df_clean



def create_derived_features(df):
    """Add new features for analysis"""
    df_clean = df.copy()
    
    # Duration in minutes
    if 'duration_ms' in df_clean.columns:
        df_clean['duration_min'] = (df_clean['duration_ms'] / 60000).round(2)
    
    # Popularity label
    if 'popularity' in df_clean.columns:
        df_clean['popularity_label'] = pd.cut(
            df_clean['popularity'], 
            bins=[0, 30, 60, 100], 
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
    
    # Energy to danceability ratio
    if 'energy' in df_clean.columns and 'danceability' in df_clean.columns:
        df_clean['energy_dance_ratio'] = (
            df_clean['energy'] / df_clean['danceability'].replace(0, 0.001)
        ).round(3)

    if 'instrumentalness' in df_clean.columns:
        df_clean['is_instrumental'] = (df_clean['instrumentalness'] > 0.5).astype(bool)
    
    if 'tempo' in df_clean.columns:
        df_clean['tempo_category'] = pd.cut(
            df_clean['tempo'],
            bins=[0, 100, 140, 300],
            labels=['Slow', 'Medium', 'Fast'],
            include_lowest=True
        )

    return df_clean

def save_clean_data(df, path="data/processed/spotify_cleaned.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Final cleaned data saved to {path}")

def preprocess_pipeline(input_path="data/raw/dataset.csv",
                        output_path="data/processed/spotify_cleaned.csv"):
   
    df = load_data(input_path)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = convert_data_types(df)
    df = create_derived_features(df)
    save_clean_data(df, output_path)
    
    return df

if __name__ == "__main__":
    df_cleaned =preprocess_pipeline()
