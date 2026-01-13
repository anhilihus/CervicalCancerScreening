import joblib
import pandas as pd
import os

def load_classification_model(model_path="model.pkl"):
    """Load the trained Random Forest model."""
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

def predict_cell_class(model, features_dict):
    """
    Predict the class of a single cell given its features.
    features_dict should contain: NC_Ratio, Circularity, Cell_Area, Nucleus_Area
    """
    if model is None:
        return "Model not loaded", 0.0

    # Ensure correct order of features matching training columns
    # ['NC_Ratio', 'Circularity', 'Cell_Area', 'Nucleus_Area', 'Perimeter']
    # Note: Perimeter was added in training script, need to include it.
    
    df = pd.DataFrame([features_dict])
    required_cols = ['NC_Ratio', 'Circularity', 'Cell_Area', 'Nucleus_Area', 'Perimeter']
    
    # Fill missing if any (shouldn't happen if pipeline is consistent)
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    df = df[required_cols]
    
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)
    
    return prediction, confidence
