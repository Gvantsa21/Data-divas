# DATA PREPROCESSING - SPOTIFY TRACK POPULARITY PREDICTION

# Notebook Objective:
# - Perform comprehensive preprocessing of the Spotify dataset before EDA or modeling
# - Steps include:
#     - Handling missing values
#     - Detecting and treating outliers
#     - Converting data types for efficiency
#     - Creating derived/engineered features
#     - Saving clean data


# 1. Setup and Imports

# Import system and path management libraries
# Append '../src' to Python path to allow importing custom preprocessing modules

# Import custom preprocessing functions from data_processing module:
# - load_data: Load CSV into DataFrame
# - handle_missing_values: Impute or fill missing data
# - handle_outliers: Detect and cap extreme values
# - convert_data_types: Optimize column data types
# - create_derived_features: Generate new useful features
# - save_clean_data: Save preprocessed DataFrame to CSV

# Import standard data libraries:
# - pandas, numpy, matplotlib.pyplot, seaborn
# Print confirmation that all imports were successful


# 2. Load Raw Data

# Load the raw dataset into a DataFrame using the custom load_data function
# Make a copy of the original dataset for comparison/validation later
# Print initial shape of the dataset (rows and columns)


# 3. Handle Missing Values

# 3.1 Check Missing Values
# - Calculate total missing values in the dataset
# - Print total missing values
# - If missing values exist, display count per column

# 3.2 Apply Missing Value Treatment
# - Use handle_missing_values function to fill or impute missing data
# - Recalculate total missing values to confirm treatment
# - Print missing values count after treatment


# 4. Outlier Detection and Treatment

# 4.1 Visualize Outliers Before Treatment
# - Select key features for outlier inspection (danceability, energy, loudness, speechiness)
# - Create a 2x2 grid of boxplots to visualize distributions and detect outliers
# - Label each plot appropriately (Before Outlier Treatment)

# 4.2 Apply Outlier Treatment
# - Use handle_outliers function to cap extreme values (IQR method)
# - Ensures extreme values do not disproportionately affect analysis or modeling

# 4.3 Visualize After Treatment
# - Repeat boxplots for the same features after outlier treatment
# - Compare distributions before and after to confirm treatment effectiveness


# 5. Data Type Conversion

# 5.1 Check Current Data Types
# - Print data types of all columns to identify optimization opportunities
# - Helps detect object types that can be converted to categorical or numeric

# 5.2 Apply Type Conversions
# - Convert appropriate columns to optimized types using convert_data_types function
# - Examples:
#     - Boolean flags -> boolean type
#     - Categorical strings -> category type
#     - Float precision optimization (float32)
# - Print data types after conversion to confirm


# 6. Feature Engineering

# 6.1 Create Derived Features
# - Count number of columns before feature creation
# - Generate new features using create_derived_features function
# - Count columns after feature creation
# - Calculate number of new features added
# - Print new features and their names

# 6.2 Verify Derived Features
# - Inspect key engineered columns:
#     - popularity_label: categorical bins (Low/Medium/High)
#     - tempo_category: Slow/Medium/Fast tempo bins
#     - valence_category: Sad/Neutral/Happy mood bins
# - Print value counts for each to confirm distribution


# 7. Sample Cleaned Data
# - Display first 10 rows of cleaned DataFrame for inspection


# 8. Save Cleaned Data

# - Save the preprocessed dataset to CSV using save_clean_data function
# - Print confirmation that data has been saved successfully


# 9. Preprocessing Summary

# - Summarize preprocessing steps and their outcomes:
#     - Initial vs final number of features
#     - New features added
#     - Missing values handled
#     - Outliers capped
#     - Data types optimized
# - Provides quick reference for reproducibility and reporting


# 10. Key Decisions and Justifications

# Missing Value Treatment:
# - Numerical features: median imputation (robust to outliers)
# - Categorical features: fill with 'Unknown' (preserve information)

# Outlier Treatment:
# - IQR-based capping (1.5 * IQR)
# - Capping retains all records while limiting extreme influence
