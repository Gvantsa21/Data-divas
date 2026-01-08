import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)

def load_data(path="data/processed/spotify_cleaned.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


def select_features(df):
    feature_cols = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo', 'duration_min'
    ]
    
    X = df[feature_cols]
    y = df['popularity_label']
    for label, count in y.value_counts().sort_index().items():
        print(f"    {label}: {count} ({count/len(y)*100:.1f}%)")
    
    return X, y, feature_cols


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)


    print(f"\n{model_name} Evaluation Metrics:")
    print(f"  accuracy:  {accuracy:.4f}")
    print(f"  precision: {precision:.4f}")
    print(f"  recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print(classification_report(y_test, y_pred))
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'y_test': y_test
    }
    return metrics


def plot_confusion_matrix(y_test, y_pred, model_name, save_path):
    cm = confusion_matrix(y_test, y_pred)
    
    labels = sorted(y_test.unique())
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, color='#D946A6', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"confusion matrix saved: {save_path}")


def plot_feature_importance(model, feature_names, save_path):
    importance_df = pd.DataFrame({'Feature': feature_names,'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], 
             color='#065F46', alpha=0.8)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance - Decision Tree', fontsize=14, 
              color='#D946A6', pad=20)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_ml_pipeline(data_path="data/processed/spotify_cleaned.csv"):
    os.makedirs('reports/figures', exist_ok=True)
    
    df = load_data(data_path)
    X, y, feature_names = select_features(df)
    
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    
    X_train_raw = X.iloc[y_train.index].values
    X_test_raw = X.iloc[y_test.index].values
    dt_model = train_decision_tree(X_train_raw, y_train)
    
    metrics_lr = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    metrics_dt = evaluate_model(dt_model, X_test_raw, y_test, "Decision Tree")
    
    
    plot_confusion_matrix(y_test, metrics_lr['y_pred'],"Logistic Regression",'reports/figures/confusion_matrix_logistic.png')
    plot_confusion_matrix(y_test, metrics_dt['y_pred'],"Decision Tree",'reports/figures/confusion_matrix_tree.png')

if __name__ == "__main__":
    run_ml_pipeline()