# EXPLORATORY DATA ANALYSIS (EDA) AND VISUALIZATION
 
# The goal here is to explore the cleaned Spotify dataset and understand patterns in the data.

# We will look at descriptive statistics, correlations, distributions, outliers, feature relationships,

# and categorical distributions. This helps us see what the data looks like before modeling.
 


# 1. Setup and Imports

# Import pandas, numpy, matplotlib, seaborn, plotly, and os.

# Define functions to load data, create plots, and analyze features.
 


# 2. Load Cleaned Data

# Load the dataset that has already been preprocessed and saved as CSV.
 


# 3. Descriptive Statistics

# Compute basic stats for numeric features: count, mean, std, min, 25%, median (50%), 75%, max.

# This helps us understand the central values and spread of features like energy, danceability, and popularity.
 


# 4. Correlation Analysis

# Look at how numeric features are related using a correlation heatmap.

# This shows which features are positively or negatively correlated.
 


# 5. Distribution Analysis

# Plot histograms and KDE plots for numeric features.

# Include mean and median lines so we can see skew or spread.
 


# 6. Outlier Analysis

# Detect outliers using the IQR method.

# Make boxplots to visually inspect extreme values and check if preprocessing worked.
 


# 7. Categorical Analysis

# Look at distributions of categories like popularity_label, tempo_category, valence_category, and top genres.

# Create bar plots to see which categories are most common.
 


# 8. Relationship Analysis

# Make scatter plots to see how numeric features relate to each other.

# Add trend lines to check if patterns are linear.

# Example: energy vs danceability, acousticness vs energy, popularity vs energy.
 


# 9. Distribution Across Categories

# Use violin plots to compare numeric features across categorical bins.

# Example: popularity for different tempo categories, energy for different valence categories.
 


# 10. Run EDA Pipeline

# The run_eda_pipeline function will:

# - Load the cleaned dataset

# - Calculate descriptive statistics

# - Run outlier analysis

# - Generate all visualizations (correlation heatmap, distributions, box plots, scatter plots, categorical bar plots, violin plots)

# - Save all plots to reports/figures folder

 
