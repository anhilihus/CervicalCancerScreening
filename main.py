import argparse
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

from src.utils import load_config, scan_images, create_output_dirs
from src.processing import CervicalCellAnalyzer
from src.visualization import plot_feature_distributions, plot_correlations

def main():
    parser = argparse.ArgumentParser(description="Cervical Cell Segmentation Pipeline")
    parser.add_argument("--input_dir", type=str, help="Input directory containing images", default=None)
    parser.add_argument("--output_dir", type=str, help="Output directory for results", default=None)
    parser.add_argument("--max_images", type=int, help="Limit number of images for testing", default=None)
    args = parser.parse_args()

    # Load Config
    config = load_config()
    
    # Override config with args if provided
    input_dir = args.input_dir if args.input_dir else config['paths']['input_dir']
    output_dir = args.output_dir if args.output_dir else config['paths']['output_dir']
    
    # Setup
    base_out, img_out, plots_out = create_output_dirs(output_dir)
    analyzer = CervicalCellAnalyzer(config)
    
    # Scan images
    print(f"Scanning images in {input_dir}...")
    image_files = scan_images(input_dir)
    print(f"Found {len(image_files)} images.")
    
    if args.max_images:
        print(f"Limiting to first {args.max_images} images.")
        image_files = image_files[:args.max_images]
    
    all_metrics = []
    
    # Process Loop
    for img_path in tqdm(image_files, desc="Processing Images"):
        # Infer category from parent folder name (e.g. NILM, HSIL)
        category = img_path.parent.name
        
        result = analyzer.process_image(img_path, category=category)
        
        if result:
            # Aggregate metrics
            all_metrics.extend(result['metrics'])
            
            # Save Annotated Image (ONLY if configured, to save time/space)
            # For massive batch runs, maybe disable this or only do it for N images.
            if config['visualization']['save_plots'] and len(image_files) < 20:
                out_name = f"{img_path.stem}_segmented.png"
                cv2.imwrite(str(img_out / out_name), result['annotated_image'])
    
    # Save CSV
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        csv_path = base_out / "segmentation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Generate Summary Plots
        print("Generating plots...")
        plot_feature_distributions(df, plots_out)
        plot_correlations(df, plots_out)
        print("Done.")
    else:
        print("No cells detected or processed.")

if __name__ == "__main__":
    main()
