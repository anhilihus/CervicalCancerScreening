import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def train_model(csv_path="output_results/segmentation_results.csv", model_out="model.pkl"):
    """
    Train a Random Forest classifier on the segmented cell data.
    """
    if not Path(csv_path).exists():
        print(f"Error: Data file {csv_path} not found. Run main.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Filter out 'Unknown' or 'Uploaded' categories if they exist in training data
    df = df[~df['Category'].isin(['Unknown', 'Uploaded'])]
    
    if len(df) == 0:
        print("Error: No labeled training data found.")
        return

    # Features and Target
    features = ['NC_Ratio', 'Circularity', 'Cell_Area', 'Nucleus_Area', 'Perimeter']
    X = df[features]
    y = df['Category']

    print(f"Training on {len(df)} cells...")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    # Using class_weight='balanced' because some classes (Carcinoma) might have fewer samples
    # Reduced n_estimators and limited max_depth to keep the model size small (<25MB)
    clf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save
    joblib.dump(clf, model_out)
    print(f"Model saved to {model_out}")

if __name__ == "__main__":
    train_model()
