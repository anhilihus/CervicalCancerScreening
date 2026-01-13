import yaml
import os
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def scan_images(directory, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    """
    Recursively scan a directory for images with specific extensions.
    Returns a list of Path objects.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(Path(root) / file)
    
    return sorted(image_files)

def create_output_dirs(base_output_dir):
    """Create necessary subdirectories for outputs."""
    base = Path(base_output_dir)
    images_dir = base / "annotated_images"
    plots_dir = base / "plots"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return base, images_dir, plots_dir
