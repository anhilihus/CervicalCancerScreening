import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import tempfile
import os
import sys

# Fix for Streamlit Cloud "ModuleNotFoundError: No module named 'src'"
# (Not needed if flattened, but harmless)
import sys
# Imports flattened
from processing import CervicalCellAnalyzer
from ml_utils import load_classification_model, predict_cell_class
# We import train_model to support server-side training
from train_model import train_model
import joblib

st.set_page_config(page_title="Cervical Cell Classifier", layout="wide")

@st.cache_resource
def get_model():
    """Load model, or train it if missing/broken."""
    model_path = "model.pkl"
    
    # Try loading
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except Exception as e:
        print(f"Model load failed ({e}). Retraining...")
        pass
        
    # If we are here, we need to train
    with st.spinner("⚠️ First time setup: Training AI Model... (this takes ~30s)"):
        # Ensure training csv exists
        import time
        t_start = time.time()
        train_model() # This saves to model.pkl
        t_end = time.time()
        st.success(f"Model trained in {t_end - t_start:.1f}s!")
        return joblib.load(model_path)

model = get_model()

st.title("🤖 AI Cervical Cancer Screening")

# Config (Auto - No Sliders)
config = {
    'thresholds': {'dark': {'min': 0, 'max': 0}, 'medium': {'min': 0, 'max': 0}}, # Ignored by auto-segmentation
    'morphology': {'kernel_size': 3, 'iterations': 1}
}

analyzer = CervicalCellAnalyzer(config)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()
    temp_path = tfile.name
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        original_image = Image.open(uploaded_file)
        st.image(original_image, use_container_width=True)

    # Process
    try:
        with st.spinner("AI is analyzing cellular structures..."):
            result = analyzer.process_image(temp_path, category="Uploaded")
        
        if result:
            with col2:
                st.subheader("AI Segmentation")
                annotated_rgb = cv2.cvtColor(result['annotated_image'], cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
            
            # Metrics & Classification
            st.divider()
            st.subheader("Diagnostic Report")
            
            metrics = result['metrics']
            if metrics:
                # Classify each cell
                classified_metrics = []
                class_counts = {}
                
                for cell in metrics:
                    if model:
                        pred_class, conf = predict_cell_class(model, cell)
                        cell['Prediction'] = pred_class
                        cell['Confidence'] = f"{conf:.2%}"
                        
                        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                    else:
                        cell['Prediction'] = "Unknown"
                        cell['Confidence'] = "0%"
                        
                    classified_metrics.append(cell)
                
                # Show Overall Result
                if class_counts:
                    # Logic: If any carcinoma detected, flag as carcinoma. Else HSIL, etc.
                    # Hierarchy of severity: Carcinoma > HSIL > LSIL > NILM
                    priority = ['carcinoma', 'HSIL', 'LSIL', 'NILM']
                    final_diagnosis = "NILM"
                    
                    found_classes = set(class_counts.keys())
                    for severity in priority:
                        # Case insensitive check
                        match = next((c for c in found_classes if c.lower() == severity.lower()), None)
                        if match:
                            final_diagnosis = match
                            break
                    
                    color = "red" if final_diagnosis.lower() in ['carcinoma', 'hsil'] else "green"
                    if final_diagnosis.lower() == 'lsil': color = "orange"
                    
                    st.markdown(f"### Overall Diagnosis: :{color}[{final_diagnosis}]")
                    
                    # Display breakdown
                    st.write("Cell Class Distribution:")
                    st.bar_chart(class_counts)

                # Table
                df = pd.DataFrame(classified_metrics)
                st.dataframe(df[['Cell_ID', 'Prediction', 'Confidence', 'NC_Ratio', 'Circularity', 'Cell_Area']])
                
            else:
                st.warning("No cells detected.")
                
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
