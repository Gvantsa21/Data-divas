import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)
import joblib


def load_data(path: str = "data/processed/spotify_cleaned.csv") -> pd.DataFrame:
    """
    Load the cleaned Spotify dataset from a CSV file.

    Parameters
    ----------
    path : str, optional
        Path to the cleaned CSV file (default is "data/processed/spotify_cleaned.csv").

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


def select_features(df: pd.DataFrame):
    """
    Select features and target variable from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    feature_cols : list
        List of feature column names
    """
    feature_cols = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness",
        "valence", "tempo", "duration_min"
    ]
    X = df[feature_cols]
    y = df["popularity_label"]
    
    # Print class distribution for awareness
    print("Class distribution:")
    for label, count in y.value_counts().sort_index().items():
        print(f"    {label}: {count} ({count / len(y) * 100:.1f}%)")
    
    return X, y, feature_cols


def split_and_scale_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into training and testing sets, then scale the features

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float, optional
        Fraction of the dataset to reserve as test set (default is 0.2)
    random_state : int, optional
        Random seed for reproducibility (default is 42)

    Returns
    -------
    X_train_scaled : np.ndarray
        Scaled training features
    X_test_scaled : np.ndarray
        Scaled test features
    X_train_raw : np.ndarray
        Raw (unscaled) training features
    X_test_raw : np.ndarray
        Raw (unscaled) test features
    y_train : pd.Series
        Training labels
    y_test : pd.Series
        Test labels
    scaler : StandardScaler
        Fitted scaler object
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features for models that require scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, X_train.values, X_test.values, y_train, y_test, scaler


def train_logistic_regression(X_train: np.ndarray, y_train: pd.Series) -> LogisticRegression:
    """
    Train a multinomial Logistic Regression model.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    model : LogisticRegression
        Trained logistic regression model.
    """
    model = LogisticRegression(max_iter=1000,random_state=42,multi_class="multinomial",solver="lbfgs")
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train: np.ndarray, y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training features (raw, not scaled).
    y_train : pd.Series
        Training labels.

    Returns
    -------
    model : DecisionTreeClassifier
        Trained decision tree model.
    """
    model = DecisionTreeClassifier(max_depth=10,min_samples_split=5,min_samples_leaf=2,random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training features (raw, not scaled).
    y_train : pd.Series
        Training labels.

    Returns
    -------
    model : RandomForestClassifier
        Trained random forest model.
    """
    model = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=5,random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: pd.Series, model_name: str):
    """
    Evaluate a trained model using standard classification metrics.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    X_test : np.ndarray
        Test features.
    y_test : pd.Series
        Test labels.
    model_name : str
        Name of the model for display.

    Returns
    -------
    metrics : dict
        Dictionary containing evaluation metrics and predictions.
    """
    y_pred = model.predict(X_test)
    
    # calculating metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    # Display results
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"  accuracy:  {accuracy:.4f}")
    print(f"  precision: {precision:.4f}")
    print(f"  recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(classification_report(y_test, y_pred))
    
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "y_pred": y_pred,
        "y_test": y_test
    }
    return metrics


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, model_name: str, save_path: str):
    """
    Plot and save a confusion matrix with custom color scheme.

    Parameters
    ----------
    y_test : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    model_name : str
        Model name to display in the plot title.
    save_path : str
        Path to save the figure.
    """
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())
    
    # Create custom color map
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#FFFFFF', '#FFB3D9', '#D946A6']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, color="#D946A6", pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"confusion matrix saved: {save_path}")


def plot_feature_importance(model, feature_names: list, save_path: str, model_name: str):
    """
    Plot and save feature importance for tree-based models.

    Parameters
    ----------
    model : sklearn estimator
        Trained tree-based model (DecisionTree or RandomForest).
    feature_names : list
        List of feature names.
    save_path : str
        Path to save the figure.
    model_name : str
        Model name to display in the plot title.
    """
    # Create importance dataframe and sort
    importance_df = pd.DataFrame({"Feature": feature_names,
                                "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    
    # Plot horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"],importance_df["Importance"],color="#065F46",alpha=0.8)
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"Feature Importance - {model_name}", fontsize=14, color="#D946A6", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"feature importance saved: {save_path}")


def plot_model_comparison(all_metrics: list, save_path: str):
    """
    Create a comparison visualization of all models across multiple metrics.

    Parameters
    ----------
    all_metrics : list
        List of dictionaries containing model evaluation metrics.
    save_path : str
        Path to save the figure.
    """
    # Convert to dataframe for easier plotting
    metrics_df = pd.DataFrame(all_metrics)
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Performance Comparison", fontsize=16, color="#D946A6", y=0.995)
    
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors = ["#D946A6", "#B4E7CE", "#FFB3D9", "#065F46"]
    
    # Plot each metric
    for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(metrics_df["model_name"], metrics_df[metric], color=colors[idx], alpha=0.8)
        ax.set_ylabel(name, fontsize=11)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"model comparison saved: {save_path}")


def run_ml_pipeline(data_path: str = "data/processed/spotify_cleaned.csv"):
    """
    Executing full machine learning pipeline:
    - Loading data
    - Selecting features
    - Spliting and scaling data
    - Training three models: Logistic Regression, Decision Tree, Random Forest
    - Evaluating models
    - Generating visualizations (confusion matrices, feature importance, comparison)
    - Save best model

    Parameters
    ----------
    data_path : str, optional
        Path to the cleaned dataset CSV.
    """
    # create output directories
    os.makedirs("reports/results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # load dataset
    df = load_data(data_path)
    
    # feature selection
    X, y, feature_names = select_features(df)
    
    # split and scale data
    X_train_scaled, X_test_scaled, X_train_raw, X_test_raw, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # training models
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    dt_model = train_decision_tree(X_train_raw, y_train)
    rf_model = train_random_forest(X_train_raw, y_train)
    
    # evaluating models
    metrics_lr = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    metrics_dt = evaluate_model(dt_model, X_test_raw, y_test, "Decision Tree")
    metrics_rf = evaluate_model(rf_model, X_test_raw, y_test, "Random Forest")
    
    all_metrics = [metrics_lr, metrics_dt, metrics_rf]
    
    # Generate confusion matrices
    plot_confusion_matrix(y_test, metrics_lr["y_pred"], "Logistic Regression", "reports/results/confusion_matrix_logistic.png")
    plot_confusion_matrix(y_test, metrics_dt["y_pred"], "Decision Tree", "reports/results/confusion_matrix_tree.png")
    plot_confusion_matrix(y_test, metrics_rf["y_pred"], "Random Forest", "reports/results/confusion_matrix_rf.png")
    
    # Generate feature importance plots
    plot_feature_importance(dt_model, feature_names, "reports/results/feature_importance_tree.png", "Decision Tree")
    plot_feature_importance(rf_model, feature_names, "reports/results/feature_importance_rf.png", "Random Forest")
    
    # Generate model comparison
    plot_model_comparison(all_metrics, "reports/results/model_comparison.png")
    
    # Printing & saving summary
    print("\n=== Summary ===")
    summary_df = pd.DataFrame(all_metrics)[["model_name", "accuracy", "precision", "recall", "f1_score"]]
    summary_df = summary_df.sort_values("f1_score", ascending=False)
    print(summary_df.to_string(index=False))

    summary_df.to_csv("reports/results/model_metrics_summary.csv", index=False)
    print(f"\nmetrics saved to: reports/results/model_metrics_summary.csv")


if __name__ == "__main__":
    run_ml_pipeline()