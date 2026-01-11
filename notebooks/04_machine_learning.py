# MACHINE LEARNING PIPELINE FOR SPOTIFY TRACK POPULARITY
 
# This script trains and evaluates three classification models (Logistic Regression, Decision Tree, Random Forest)

# to predict Spotify track popularity (Low, Medium, High). It also generates confusion matrices,

# feature importance plots, and compares model performance.


 
# 1. Setup and Imports

# Import all necessary libraries: pandas, numpy, matplotlib, seaborn, sklearn models and metrics.

# Functions include: loading data, selecting features, splitting/scaling, training models, evaluation, plotting.
 


# 2. Load and Prepare Data

# Load the cleaned Spotify dataset CSV.

# Select audio features as predictors and popularity_label as the target.

# Print class distribution to see if classes are balanced.

# Split data into training and test sets (80-20) with stratification.

# Scale numerical features for Logistic Regression, keep raw for tree-based models.
 


# 3. Train Models

# Train three models:

# - Logistic Regression (linear, baseline)

# - Decision Tree (non-linear, interpretable)

# - Random Forest (ensemble, usually best performance)
 


# 4. Evaluate Models

# Use test data to calculate accuracy, precision, recall, and F1 score.

# Print classification report for each model.

# Weighted average is used to handle class imbalance.
 


# 5. Confusion Matrices

# Plot confusion matrices for all three models to see where misclassifications occur.

# Save plots to reports/results folder.
 


# 6. Feature Importance

# For tree-based models (Decision Tree, Random Forest):

# - Plot horizontal bar charts showing the top features influencing predictions.

# - Identify which audio features matter most for popularity.
 


# 7. Model Comparison

# Compare all models on accuracy, precision, recall, and F1 score.

# Generate a bar chart to visualize differences and save to reports/results.
 


# 8. Best Model Selection

# Select the model with the highest F1 score.

# Print performance metrics and save summary table as CSV.
 


# 9. Summary of Findings

# Logistic Regression: fast, baseline, lowest performance.

# Decision Tree: moderate performance, interpretable, may overfit.

# Random Forest: highest F1 and accuracy, handles feature interactions well.

# Most important features for predicting popularity: Energy, Danceability, Loudness, Acousticness, Valence.

# Popularity prediction is moderate because other factors (artist, marketing) also affect popularity.
 

  
# 10. Run Pipeline

# The run_ml_pipeline() function executes the whole pipeline end-to-end:

# - Load and prepare data

# - Train and evaluate models

# - Generate plots

 