# DATA PREPROCESSING - SPOTIFY TRACK POPULARITY PREDICTION
 
# goal here is to Perform comprehensive preprocessing of the Spotify dataset before EDA or modeling
# - Steps include:
#     - Loading and sampling the dataset
#     - Handling missing values
#     - Detecting and treating outliers
#     - Converting data types for efficiency
#     - Creating derived/engineered features
#     - Validating and saving clean data
 

# 1. Setup and Imports
# Import standard data libraries:
# - pandas, numpy, matplotlib.pyplot, seaborn
 

# 2. Load and Sample Raw Data
# Load the raw dataset using custom load_data function
# Function automatically:
# - Reads CSV from data/raw/dataset.csv
# - Prints dataset info (114,000 rows × 21 columns)
# - Shows data types and memory usage
# - Displays missing values per column (1 each in artists, album_name, track_name)
# - Returns DataFrame

# Sample dataset to 60,000 rows for optimal visualization performance
# - Use random_state=42 for reproducibility
# - Reduces processing time while maintaining data richness
 

# 3. Handle Missing Values
#  Inspect Missing Values Before Treatment
# - Calculate total missing values in the dataset
# - Display missing values per column
# - Expected: 1 missing value each in 'artists', 'album_name', 'track_name'

# Apply Missing Value Treatment
# Use handle_missing_values function to fill or impute missing data
# Strategy:
# - Categorical columns (artists, album_name, track_name): Fill with 'Unknown'
# - Numerical columns: Fill with median (robust to outliers)
# Function prints: "Filling X missing values in 'column' with 'Unknown'" or median value

#Verify No Missing Values Remain
# - Recalculate total missing values
# - Confirm all missing values have been handled
 

# 4. Outlier Detection and Treatment
# Visualize Outliers Before Treatment
# - Select key features for outlier inspection: danceability, energy, loudness, speechiness
# - Create 2×2 grid of boxplots to visualize distributions
# - Label plots: "Distribution Before Outlier Treatment"
# - Helps identify extreme values that need treatment

# Apply Outlier Treatment
# Use handle_outliers function to treat extreme values
# Method: IQR-based capping (1.5 × IQR)
# - Calculates Q1 (25th percentile) and Q3 (75th percentile)
# - Computes IQR = Q3 - Q1
# - Sets bounds: lower = Q1 - 1.5×IQR, upper = Q3 + 1.5×IQR
# - CAPS outliers to bounds (does not remove them)
# - Preserves dataset size (60,000 rows maintained)
# Excluded columns: id, popularity, duration_ms, key, mode, time_signature, explicit
# Applied to: danceability, energy, loudness, speechiness, acousticness, 
#             instrumentalness, liveness, valence, tempo
# Function prints: "Outliers capped to IQR bounds" and shape confirmation
 
# Visualize After Treatment
# - Repeat boxplots for same features after outlier treatment
# - Compare distributions before and after to confirm treatment effectiveness
# - Should show reduced extreme values while maintaining data shape


# 5. Data Type Conversion
# Check Current Data Types
# - Print data types of all columns before conversion
# - Calculate memory usage before optimization
# - Identifies columns that can be optimized (object → category, float64 → float32)
 
# Apply Type Conversions
# Use convert_data_types function to optimize data types
# Conversions applied:
# - Boolean: 'explicit' → bool type (already converted during load)
# - Categorical: artists, album_name, track_name, track_genre → category type
# - Float precision: danceability, energy, loudness, speechiness, acousticness, 
#                   instrumentalness, liveness, valence, tempo → float32
#
# Reduced memory usage (category type uses less memory for repeated strings)
# Faster operations on categorical data
# Lower precision float uses half the memory of float64
 
# Verifying Type Conversions
# - Print data types after conversion
# - Calculate memory usage after optimization
# - Compare before/after to confirm memory savings
 

# 6. Feature Engineering
#  Create Derived Features
# Count initial number of features (21 original columns)
# Use create_derived_features function to generate 7 new features:
# 
# duration_min: Convert duration_ms to minutes for readability
#    - Formula: duration_ms / 60000, rounded to 2 decimals
# 
# popularity_label: Categorical bins for classification target
#    - Bins: [0-30: 'Low', 30-60: 'Medium', 60-100: 'High']
#    - Uses pd.cut with include_lowest=True
# 
# energy_dance_ratio: Interaction feature
#    - Formula: energy / danceability (replace 0 with 0.001 to avoid division errors)
#    - Rounded to 3 decimals
#    - Captures relationship between energy and danceability
# 
# is_instrumental: Binary indicator for instrumental tracks
#    - Boolean: instrumentalness > 0.5
# 
# tempo_category: Tempo bins for pacing analysis
#    - Bins: [0-100: 'Slow', 100-140: 'Medium', 140-300: 'Fast']
# 
# is_acoustic: Binary indicator for acoustic tracks
#    - Boolean: acousticness > 0.5
# 
# valence_category: Mood/positiveness bins
#    - Bins: [0-0.33: 'Sad', 0.33-0.67: 'Neutral', 0.67-1.0: 'Happy']
#    - Tracks emotional tone of music
# 
# Count final number of features (28 total = 21 original + 7 new)
# Calculate number of new features added
 
# Verifying Derived Features
# Inspect distributions of categorical derived features:
# - popularity_label: Check Low/Medium/High distribution
# - tempo_category: Check Slow/Medium/Fast distribution
# - valence_category: Check Sad/Neutral/Happy distribution
# - is_instrumental: Check True/False counts
# - is_acoustic: Check True/False counts
# Use value_counts() to display frequency distributions
# Ensures features were created correctly and have reasonable distributions


# 7. Data Validation
# Check for Remaining Issues
# - Verify missing values = 0 after treatment
# - Count duplicate rows (should be 0 or minimal)
# - Confirm final dataset shape: (60000, 28)
 
# Verify Audio Feature Ranges
# Check that audio features are within expected [0, 1] range:
# - Features to check: danceability, energy, speechiness, acousticness, 
#                     instrumentalness, liveness, valence
# - Print min and max for each feature
# - Flag any values outside [0, 1] as potential issues
# - After IQR capping, all should be within valid range
 
# Sample Cleaned Data
# - Display first 10 rows of cleaned DataFrame
# - Inspect both original and derived features
# - Confirm all preprocessing steps applied correctly
# - Display summary statistics for key numerical features


# Save Cleaned Data
# Save preprocessed dataset to CSV using save_clean_data function
# - Output path: data/processed/spotify_cleaned.csv
# - Creates output directory if it doesn't exist
# - Saves without index column
# Function prints: "Final cleaned data saved to data/processed/spotify_cleaned.csv"
# This cleaned dataset will be used for EDA and ML modeling
 

# Summary of Preprocessing Steps
# Display comprehensive summary of preprocessing pipeline:
# - Original records: 114,000 (full dataset)
# - Sampled records: 60,000 (for performance)
# - Final records: 60,000 (preserved after capping outliers)
# - Initial features: 21
# - Final features: 28
# - New features added: 7
# - Missing values handled: 3 (1 in each of 3 categorical columns)
# - Outliers treated: Capped using IQR method across 9 audio features
# - Data types optimized: category, float32, bool
# - Output file location
 


# 10. Key Preprocessing Decisions
# SAMPLING DECISION:
# - Reduced from 114K to 60K records
# - Reason: Optimal balance between data richness and visualization/processing performance
# - Method: Random sampling with random_state=42 for reproducibility
# - Result: Faster EDA and ML training without significant information loss
 
# final cleaned dataset saved to data/processed/spotify_cleaned.csv
