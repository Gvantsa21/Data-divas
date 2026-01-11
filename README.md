# Spotify Track Popularity Prediction
 
## Team Members
 
- Ani Kharabadze
- Mariam Phirtskhalava
- Gvantsa Tchuradze

 
## Problem Statement
 
This project analyzes Spotify track data to predict track popularity based on audio features. The goal is to identify which musical characteristics contribute most to a track's popularity and develop machine learning models capable of predicting popularity categories (Low, Medium, High).
 
## Objectives
 
- Perform comprehensive exploratory data analysis on Spotify track features
- Identify patterns and correlations between audio characteristics and popularity
- Build and compare multiple machine learning classification models
- Provide actionable insights for understanding track popularity factors
 
## Dataset Description
 
**Source:** [Spotify Tracks Dataset - Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?resource=download)
 
**Size:** 60,000 tracks
 
**Features:**
- Audio characteristics: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
- Track metadata: track name, artist, album, genre, duration
- Target variable: popularity (0-100), categorized as Low/Medium/High
 
**Preprocessing:**
- Missing values handled via median/mode imputation
- Outliers capped using IQR method
- Derived features created: `duration_min`, `popularity_label`, `tempo_category`, `valence_category`
 
## Installation and Setup
 
### Prerequisites
 
- Python 3.8 or higher
- pip package manager
 
### Installation Steps
 
1. Clone the repository:
```bash
git clone <repository-url>
cd spotify-track-analysis
Create virtual environment:
 
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
 
bash
Copy code
pip install -r requirements.txt
Verify installation:
 
bash
Copy code
python -c "import pandas; import sklearn; print('Installation successful')"
Project Structure
powershell
Copy code
spotify-track-analysis/
├── data/
│   ├── raw/                    # Original dataset files
│   └── processed/              # Cleaned data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_eda_visualization.ipynb
│   └── 04_machine_learning.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Data cleaning functions
│   ├── eda.py                  # EDA functions
│   └── models.py               # ML implementations
├── reports/
│   ├── figures/                # Visualizations
│   └── results/                # Model outputs
├── models/                     # Saved models
├── app.py                      # Interactive dashboard
├── README.md
├── CONTRIBUTIONS.md
├── requirements.txt
└── .gitignore
Usage
Data Processing
Run the preprocessing pipeline:
 
bash
Copy code
python src/data_processing.py
This will:
 
Load raw data from data/raw/
 
Clean and preprocess the dataset
 
Save cleaned data to data/processed/spotify_cleaned.csv
 
Exploratory Data Analysis
Run EDA analysis:
 
bash
Copy code
python src/eda.py
Generates:
 
Distribution plots
 
Correlation heatmaps
 
Box plots for outlier detection
 
Categorical analysis
 
Scatter plots with trend lines
 
Violin plots
 
Interactive visualizations
 
Machine Learning
Train and evaluate models:
 
bash
Copy code
python src/models.py
Trains three classification models:
 
Logistic Regression
 
Decision Tree
 
Random Forest
 
Outputs:
 
Model performance metrics
 
Confusion matrices
 
Feature importance plots
 
Model comparison chart
 
Best model saved to models/best_model.pkl
 
Interactive Dashboard
Launch the Streamlit dashboard:
 
bash
Copy code
streamlit run app.py
Features:
 
Interactive data exploration
 
Real-time model predictions
 
3D visualizations
 
Parallel coordinates plot
 
Results Summary
Model Performance
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	67.7%	67.1%	67.7%	66.9%
Decision Tree	70.8%	71.5%	70.8%	70.4%
Random Forest	75.4%	76.1%	75.4%	75.2%
 
Best Model: Random Forest with 75.4% accuracy
 
Key Insights
Energy and Loudness show the strongest positive correlation (r = 0.72)
 
Acousticness and Energy exhibit strong negative correlation (r = -0.65)
 
Popularity shows weak correlation with individual audio features, suggesting it depends on multiple factors
 
Tempo has minimal impact on popularity across all categories
 
Valence (mood) distribution is skewed toward neutral and sad tracks
 
Feature Importance
Top 5 most important features for predicting popularity:
 
Energy
 
Danceability
 
Loudness
 
Acousticness
 
Valence
 
Methodology
Data Preprocessing
Handled missing values using median imputation for numerical features
 
Detected and capped outliers using IQR method (1.5 × IQR)
 
Created derived features: popularity categories, tempo categories, mood categories
 
Standardized features for model training
 
Exploratory Data Analysis
Generated 7+ visualization types
 
Performed correlation analysis on 10 audio features
 
Conducted outlier analysis across all numerical variables
 
Analyzed categorical distributions
 
Machine Learning
Implemented three classification models
 
Used 80/20 train-test split with stratification
 
Evaluated using accuracy, precision, recall, and F1 score
 
Compared models using multiple metrics
 
Selected best model based on F1 score
 
Limitations
Dataset limited to acoustic genre tracks
 
Popularity metric may be influenced by temporal factors not captured
 
Model performance could improve with additional features (release date, artist popularity)
 
External factors (marketing, playlisting) not included in analysis
 
Future Work
Incorporate temporal features (release date, trending status)
 
Expand to multiple genres for broader applicability
 
Implement deep learning models for improved accuracy
 
Add real-time prediction API
 
Integrate user listening history data
 
Technical Details
Languages: Python 3.8+
Libraries:
 
Data Processing: pandas, numpy
 
Visualization: matplotlib, seaborn, plotly
 
Machine Learning: scikit-learn
 
Dashboard: streamlit
 
Development Tools:
 
Jupyter Notebook for exploratory analysis
 
Git for version control
 
Virtual environment for dependency management
 
References
Dataset: Spotify Tracks Dataset - Kaggle
 
Scikit-learn Documentation: https://scikit-learn.org
 
Pandas Documentation: https://pandas.pydata.org
 
Project Guidelines: Data Science with Python Final Project
 
License
This project is submitted as part of the Data Science with Python course.
 
Contact
For questions or feedback, please contact:
 
 
 
Mariam Phirtskhalava
 
Ani Kharabadze
 
Gvantsa Tchuradze
 
Date: January 2026
Course: Data Science with Python
Institution: Kutaisi International University 
