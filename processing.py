import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

class CervicalCellAnalyzer:
    def __init__(self, config):
        self.config = config
        self.dark_thresh = (config['thresholds']['dark']['min'], config['thresholds']['dark']['max'])
        self.med_thresh = (config['thresholds']['medium']['min'], config['thresholds']['medium']['max'])
        self.results = []

    def process_image(self, image_path, category="Unknown"):
        """
        Run the full segmentation and analysis pipeline on a single image.
        """
        image_path = str(image_path)
        original_color = cv2.imread(image_path)
        if original_color is None:
            print(f"Error: Could not read image {image_path}")
            return None

        # 1. Preprocessing
        gray = cv2.cvtColor(original_color, cv2.COLOR_BGR2GRAY)
        
        # Histogram Equalization
        equalized = cv2.equalizeHist(gray)
        
        # Gaussian Blur to remove noise
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

        # 2. Segmentation using Auto-Thresholding (Otsu's Method)
        
        # A. Nucleus Detection
        # Nuclei are the darkest parts. Inverted thresholding with Otsu handles this well.
        # However, for better separation, we can also use adaptive thresholding.
        # Let's try Otsu on the blurred image.
        _, nuclei_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # B. Cytoplasm Detection
        # Cytoplasm is lighter than nucleus but darker than background.
        # Standard Otsu might pick up both. 
        # A common trick is to use a secondary threshold or simply say anything "not background".
        # Let's assume background is very bright (near 255).
        # We can use a high constant threshold for "cell vs background" since background is usually white-ish in Pap smears.
        # Or, we can use Triangle thresholding which is good for histograms with a peak at one end (background).
        _, cell_body_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
        
        # Refine Masks
        nuclei_mask = nuclei_thresh
        cytoplasm_mask = cell_body_thresh
        
        # Ensure nucleus is part of the cell (logical AND) - though topologically it should be inside.
        # But we want the cytoplasm mask to represent the WHOLE cell (including nucleus) for contour finding.
        # So 'cell_body_thresh' is good.

        # Morphological Operations to clean up
        kernel = np.ones((self.config['morphology']['kernel_size'], self.config['morphology']['kernel_size']), np.uint8)
        nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        cytoplasm_mask = cv2.morphologyEx(cytoplasm_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cytoplasm_mask = cv2.morphologyEx(cytoplasm_mask, cv2.MORPH_CLOSE, kernel, iterations=2) # More closing to fill gaps

        # 3. Contour Detection & Feature Extraction
        # We find contours for cells (cytoplasm) and then look for nuclei inside or near them
        # Note: This is a simplified logic. In reality, cytoplasm often encompasses the nucleus. 
        # The report suggests finding 'dark' and 'medium' regions. 
        # We will treat 'cytoplasm_mask' as the cell body roughly, and 'nuclei_mask' as nucleus.
        
        cell_contours, _ = cv2.findContours(cytoplasm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nuclei_contours, _ = cv2.findContours(nuclei_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated_img = original_color.copy()
        
        # Analyze each potential cell (using cytoplasm contours as the 'cell' boundary)
        # Filter small noise
        min_area = 50
        valid_cells = [c for c in cell_contours if cv2.contourArea(c) > min_area]
        
        image_metrics = []

        for i, cell_cnt in enumerate(valid_cells):
            cell_area = cv2.contourArea(cell_cnt)
            cell_perimeter = cv2.arcLength(cell_cnt, True)
            if cell_perimeter == 0: continue
            
            circularity = (4 * np.pi * cell_area) / (cell_perimeter ** 2)
            
            # Find nucleus within this cell
            # Creates a mask for the current cell
            mask_curr_cell = np.zeros_like(gray)
            cv2.drawContours(mask_curr_cell, [cell_cnt], -1, 255, -1)
            
            # Find intersection with nuclei mask
            # The intersection gives the nucleus area within this specific cell
            nucleus_region = cv2.bitwise_and(nuclei_mask, nuclei_mask, mask=mask_curr_cell)
            
            # We can calculate nucleus area just by counting non-zero pixels in the intersection
            nucleus_area = cv2.countNonZero(nucleus_region)
            
            # NC Ratio
            # Avoid division by zero. If cell_area is nucleus_area (unlikely), ratio is 1.
            # Usually Cytoplasm Area = Cell Area - Nucleus Area, or just ratio of Nucleus/Cell Area
            # The paper mentions "Nucleus-to-Cytoplasm Ratio". 
            # Often N/C = Nucleus Area / Cytoplasm Area. 
            # Cytoplasm Area here would be Total Cell Area - Nucleus Area.
            
            cytoplasm_area = max(cell_area - nucleus_area, 1.0) # avoid 0
            nc_ratio = nucleus_area / cytoplasm_area

            # Store metrics
            cell_id = f"cell_{i}"
            image_metrics.append({
                "Image": Path(image_path).name,
                "Category": category,
                "Cell_ID": cell_id,
                "Cell_Area": cell_area,
                "Nucleus_Area": nucleus_area,
                "NC_Ratio": nc_ratio,
                "Circularity": circularity,
                "Perimeter": cell_perimeter
            })

            # Annotation
            # Draw Cell (Green)
            cv2.drawContours(annotated_img, [cell_cnt], -1, (0, 255, 0), 2)
            
            # Draw Nucleus (Red) - we need contours of the nucleus region we found
            n_cnts, _ = cv2.findContours(nucleus_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_img, n_cnts, -1, (0, 0, 255), 2)

        # 4. Quality Assessment (PSNR, SSIM)
        # Compare Gray original vs Gray Processed (Annotated converted back to gray for structural check? 
        # OR usually Original vs Reconstructed. The report compares "Original" vs "Segmented".
        # Let's compare Original Gray vs Equalized/Blurred to show preprocessing fidelity, 
        # or Original vs Annotated (which will be very different).
        # The paper says "PSNR: Ratio between max power of signal (original) and noise (error introduced)".
        # Usually checking if preprocessing (hist eq + blur) destroyed info.
        
        psnr_val = cv2.PSNR(gray, blurred)
        ssim_val = ssim(gray, blurred, data_range=blurred.max() - blurred.min())

        return {
            "metrics": image_metrics,
            "annotated_image": annotated_img,
            "quality": {"PSNR": psnr_val, "SSIM": ssim_val}
        }
