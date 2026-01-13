import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_feature_distributions(df, output_dir):
    """Plot histograms/boxplots for key features."""
    output_dir = Path(output_dir)
    
    # Set style
    sns.set(style="whitegrid")
    
    # 1. NC Ratio Distribution by Category
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Category", y="NC_Ratio", data=df)
    plt.title("Nucleus-to-Cytoplasm Ratio by Category")
    plt.savefig(output_dir / "nc_ratio_distribution.png")
    plt.close()
    
    # 2. Circularity Distribution by Category
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Category", y="Circularity", data=df)
    plt.title("Cell Circularity by Category")
    plt.savefig(output_dir / "circularity_distribution.png")
    plt.close()

def plot_correlations(df, output_dir):
    """Plot scatter plots for feature correlations."""
    output_dir = Path(output_dir)
    sns.set(style="whitegrid")
    
    # Scatter: Cell Area vs Nucleus Area
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Cell_Area", y="Nucleus_Area", hue="Category", data=df, alpha=0.7)
    plt.title("Correlation: Cell Area vs Nucleus Area")
    plt.savefig(output_dir / "area_correlation.png")
    plt.close()
