import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
 
def load_cleaned_data(path="data/processed/spotify_cleaned.csv"):
    """
    Loading preprocessed dataset from CSV file
    Parameters
    ----------
    path : str, optional
        Path to the cleaned CSV file
   
    Returns
    -------
    pd.DataFrame
        Loaded dataset
   
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df
 
 
def descriptive_statistics(df):
    """
    computing descriptive statistics
   
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
   
    Returns
    -------
    pd.DataFrame
        summary, including mean, std, min, max, quartiles, median
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numerical_cols].describe()
    stats.loc['median'] = df[numerical_cols].median()
    print("\nDescriptive Statistics:\n", stats)
    return stats
 
 
def correlation_analysis(df):
    """
    correlation heatmap for audio features
   
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with audio features
   
    Returns
    -------
    nothing
        Saves heatmap to reports
    """
    audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                      'acousticness', 'instrumentalness', 'liveness',
                      'valence', 'tempo', 'popularity']
   
    corr_matrix = df[audio_features].corr()
 
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=['#065F46', '#B4E7CE', '#FFB3D9', '#D946A6'], center=0, square=True, linewidths=1)
    plt.title('Correlation Heatmap of Audio Features', fontsize=16, pad=20, color='#D946A6')
    plt.tight_layout()
 
    plt.savefig('reports/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Correlation heatmap saved.")
 
 
def distribution_plots(df):
    """
    creating distribution plots with mean and median lines
   
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
   
    Returns
    -------
    nothing
        Saves distribution plots
    """
    features = ['popularity', 'danceability', 'energy', 'valence']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    colors = ['#D946A6', '#065F46', '#FFB3D9', '#B4E7CE']
 
    #histogram with KDE
    for idx, feature in enumerate(features):
        sns.histplot(df[feature], kde=True, ax=axes[idx], color=colors[idx], bins=30)
        axes[idx].set_title(f'Distribution of {feature.capitalize()}', fontsize=12, color='#D946A6')
   
        mean_val = df[feature].mean()
        median_val = df[feature].median()
 
        axes[idx].axvline(mean_val, color='#065F46', linestyle='--', label=f'Mean: {mean_val:.2f}')
        axes[idx].axvline(median_val, color='#FFB3D9', linestyle='--', label=f'Median: {median_val:.2f}')
        axes[idx].set_xlabel(feature.capitalize())
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
   
    plt.tight_layout()
    plt.savefig('reports/figures/distribution_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Distribution plots saved.")
 
 
def box_plots_analysis(df):
    """
    box plots for detecting outliers in audio features
   
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
   
    Returns
    -------
        saves box plots
    """
    features = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'valence', 'tempo']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
   
    for idx, feature in enumerate(features):
        sns.boxplot(y=df[feature], ax=axes[idx], color='#B4E7CE')
        axes[idx].set_title(f'{feature.capitalize()}', color='#D946A6')
        axes[idx].set_ylabel('')
 
    # removing unused subplots
    if len(features) < len(axes):
        for i in range(len(features), len(axes)):
            fig.delaxes(axes[i])
   
    plt.suptitle('Box Plots for Audio Features', fontsize=16, y=1.02, color='#D946A6')
    plt.tight_layout()
    plt.savefig('reports/figures/box_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Box plots saved.")
 
 
def categorical_analysis(df):
    """
   e categorical feature distributions visualisation
   
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with categorical features
   
    Returns
    -------
    none
        saves categorical analysis
    """
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 
    #popularity distribution
    popularity_counts = df['popularity_label'].value_counts()
    axes[0, 0].bar(popularity_counts.index, popularity_counts.values, color='#FFB3D9')
    axes[0, 0].set_title('Popularity Distribution', color='#D946A6')
    axes[0, 0].set_xlabel('Popularity Category')
    axes[0, 0].set_ylabel('Count')
 
    #tempo distribution
    tempo_counts = df['tempo_category'].value_counts()
    axes[0, 1].bar(tempo_counts.index, tempo_counts.values, color='#B4E7CE')
    axes[0, 1].set_title('Tempo Distribution', color='#D946A6')
    axes[0, 1].set_xlabel('Tempo Category')
    axes[0, 1].set_ylabel('Count')
   
    #valence distribution
    valence_counts = df['valence_category'].value_counts()
    axes[1, 0].bar(valence_counts.index, valence_counts.values, color='#065F46')
    axes[1, 0].set_title('Valence (Mood) Distribution', color='#D946A6')
    axes[1, 0].set_xlabel('Valence Category')
    axes[1, 0].set_ylabel('Count')
   
    #top 10 genres
    top_genres = df['track_genre'].value_counts().head(10)
    axes[1, 1].barh(top_genres.index, top_genres.values, color='#D946A6')
    axes[1, 1].set_title('Top 10 Genres', color='#D946A6')
    axes[1, 1].set_xlabel('Count')
    axes[1, 1].invert_yaxis()
   
    plt.tight_layout()
    plt.savefig('reports/figures/categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Categorical analysis saved.")
 
 
def scatter_plots_relationships(df):
    """
    creating scatter plots with trend lines to show relationships between features
   
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
   
    Returns
    -------
    None
        Saves scatter plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#065F46', '#FFB3D9', '#B4E7CE', '#D946A6']
   
    #energy vs danceability
    axes[0, 0].scatter(df['danceability'], df['energy'], alpha=0.3, s=10, color=colors[0])
    axes[0, 0].set_xlabel('Danceability')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].set_title('Energy vs Danceability', color='#D946A6')
    z = np.polyfit(df['danceability'], df['energy'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['danceability'], p(df['danceability']), color='#D946A6', linestyle='--', alpha=0.8)
   
    #popularity vs energy
    axes[0, 1].scatter(df['energy'], df['popularity'], alpha=0.3, s=10, color=colors[1])
    axes[0, 1].set_xlabel('Energy')
    axes[0, 1].set_ylabel('Popularity')
    axes[0, 1].set_title('Popularity vs Energy', color='#D946A6')
    z = np.polyfit(df['energy'], df['popularity'], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(df['energy'], p(df['energy']), color='#065F46', linestyle='--', alpha=0.8)
   
    #acousticness vs energy
    axes[1, 0].scatter(df['acousticness'], df['energy'], alpha=0.3, s=10, color=colors[2])
    axes[1, 0].set_xlabel('Acousticness')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('Energy vs Acousticness', color='#D946A6')
    z = np.polyfit(df['acousticness'], df['energy'], 1)
    p = np.poly1d(z)
    axes[1, 0].plot(df['acousticness'], p(df['acousticness']), color='#D946A6', linestyle='--', alpha=0.8)
   
    #valence vs danceability
    axes[1, 1].scatter(df['valence'], df['danceability'], alpha=0.3, s=10, color=colors[3])
    axes[1, 1].set_xlabel('Valence')
    axes[1, 1].set_ylabel('Danceability')
    axes[1, 1].set_title('Danceability vs Valence', color='#D946A6')
    z = np.polyfit(df['valence'], df['danceability'], 1)
    p = np.poly1d(z)
    axes[1, 1].plot(df['valence'], p(df['valence']), color='#065F46', linestyle='--', alpha=0.8)
   
    plt.tight_layout()
    plt.savefig('reports/figures/scatter_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Scatter plots saved.")
 
 
def violin_plots(df):
    """
    violin plots to show distribution across categories    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with categorical features
   
    Returns
    -------
    nothing
        saves violin plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
   
    # popularity by tempo category
    sns.violinplot(data=df, x='tempo_category', y='popularity', ax=axes[0], palette=['#FFB3D9', '#B4E7CE', '#065F46'])
    axes[0].set_title('Popularity Distribution by Tempo Category', color='#D946A6')
    axes[0].set_xlabel('Tempo Category')
    axes[0].set_ylabel('Popularity')
   
    # energy by valence category
    sns.violinplot(data=df, x='valence_category', y='energy', ax=axes[1], palette=['#D946A6', '#FFB3D9', '#B4E7CE'])
    axes[1].set_title('Energy Distribution by Valence Category', color='#D946A6')
    axes[1].set_xlabel('Valence Category')
    axes[1].set_ylabel('Energy')
   
    plt.tight_layout()
    plt.savefig('reports/figures/violin_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Violin plots saved.")
 
 
   
def outlier_analysis(df):
    """
    Identifying outliers using IQR method
   
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
   
    Returns
    -------
    None
        Prints outlier statistics
    """
    features = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'valence', 'tempo', 'popularity']
   
    print("\n=== Outlier Analysis ===")
 
    #outliers for each feature
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
       
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_pct = (len(outliers) / len(df)) * 100
       
        print(f"{feature}: {len(outliers)} outliers ({outlier_pct:.2f}%)")
        if len(outliers) > 0:
            print(f"  Range: [{df[feature].min():.2f}, {df[feature].max():.2f}]")
            print(f"  Outlier bounds: <{lower_bound:.2f} or >{upper_bound:.2f}\n")
 
 
def interactive_time_features(df, save_path="reports/figures/interactive_duration_tempo.html"):
    """
    interactive scatter showing duration vs tempo relationship
   
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    save_path : str
        Path to save HTML file
    """
    # Sample and filter extreme outliers for better visualization
    sample_df = df[(df['duration_min'] < 15) & (df['tempo'] < 200)].sample(min(8000, len(df)))
   
    fig = px.scatter(
        sample_df,
        x='duration_min',
        y='tempo',
        color='tempo_category',
        color_discrete_map={'Slow': '#B4E7CE', 'Medium': '#FFB3D9', 'Fast': '#D946A6'},
        hover_data=['track_name', 'artists', 'popularity'],
        title='Track Duration vs Tempo - Interactive Exploration',
        labels={'duration_min': 'Duration (minutes)', 'tempo': 'Tempo (BPM)'},
        opacity=0.6
    )
   
    fig.update_layout(
        title_font_color='#D946A6',
        title_font_size=16
    )
   
    fig.write_html(save_path)
    print(f"Interactive duration vs tempo plot saved: {save_path}")
 
 
def run_eda_pipeline(data_path="data/processed/spotify_cleaned.csv"):
    """
    executes complete EDA pipeline including all analyses and visualizations
   
    Parameters
    ----------
    data_path : str, optional
        Path to cleaned dataset
   
    Returns
    -------
        Generates all visualizations and statistical analyses
    """
    os.makedirs('reports/figures', exist_ok=True)
   
    df = load_cleaned_data(data_path)
   
    descriptive_statistics(df)
    outlier_analysis(df)
   
    #visualizations
    correlation_analysis(df)
    distribution_plots(df)
    box_plots_analysis(df)
    categorical_analysis(df)
    scatter_plots_relationships(df)
    violin_plots(df)
    interactive_time_features(df)
   
    print("\nEDA complete - all visualizations saved to reports/figures/")
 
 
if __name__ == "__main__":
    run_eda_pipeline()