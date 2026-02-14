import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Voronoi
from scipy.io import loadmat
from sklearn.cluster import DBSCAN
import os
import json
import glob
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
BASE_PATH = 'ShanghaiTech'
OUTPUT_ROOT = 'final_model_dataset'

# --- TUNING PARAMETERS ---
CRITICAL_AREA_THRESHOLD = 250  # Pixels. Smaller = Stricter pressure definition.
DBSCAN_EPS = 50 
DBSCAN_MIN_SAMPLES = 5

# Risk Levels (Count of crushed people per grid cell)
# Adjust these numbers based on how strict you want to be.
RISK_THRESHOLDS = { 'LOW': 5, 'MEDIUM': 15, 'HIGH': 30, 'CRITICAL': 50 }

# Colors for Visualization
RISK_COLORS = { 
    'LOW': '#FFD700',       # Gold
    'MEDIUM': '#FF8C00',    # Orange
    'HIGH': '#FF0000',      # Red
    'CRITICAL': '#8B0000'   # Dark Red
}

# --- HELPER FUNCTIONS ---
def polygon_area(vertices):
    """Calculates polygon area using Shoelace formula."""
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_grid_coords(x, y, img_w, img_h):
    """Maps an (x,y) pixel to a 10x10 grid coordinate."""
    col_w = img_w / 10
    row_h = img_h / 10
    col_idx = int(max(0, min(x // col_w, 9)))
    row_idx = int(max(0, min(y // row_h, 9)))
    
    # Grid Label (e.g., E5)
    grid_label = f"{chr(ord('A') + col_idx)}{row_idx}"
    return col_idx, row_idx, grid_label

def get_risk_level(count):
    if count >= RISK_THRESHOLDS['CRITICAL']: return 'CRITICAL'
    if count >= RISK_THRESHOLDS['HIGH']: return 'HIGH'
    if count >= RISK_THRESHOLDS['MEDIUM']: return 'MEDIUM'
    if count >= RISK_THRESHOLDS['LOW']: return 'LOW'
    return None

def process_folder(part_name, input_folder_name, output_split_name):
    """
    part_name: 'part_A' or 'part_B'
    input_folder_name: 'train_data', 'validation_data', or 'test_data'
    output_split_name: 'train', 'val', or 'test'
    """
    
    # 1. Setup Inputs
    # Note: validation_data might not have 'images' subdir if we just moved files directly.
    # We check both standard structures.
    base_in = os.path.join(BASE_PATH, part_name, input_folder_name)
    
    if os.path.exists(os.path.join(base_in, 'images')):
        img_dir = os.path.join(base_in, 'images')
        gt_dir = os.path.join(base_in, 'ground-truth')
    else:
        # Fallback if validation_data has files directly in root
        img_dir = base_in
        gt_dir = os.path.join(base_in, 'ground-truth') # Assuming GT is always in a subfolder or parallel

    if not os.path.exists(img_dir):
        print(f"   [Skipped] {input_folder_name} not found in {part_name}")
        return

    # 2. Setup Outputs
    save_vis_dir = os.path.join(OUTPUT_ROOT, output_split_name, 'visualizations')
    save_lbl_dir = os.path.join(OUTPUT_ROOT, output_split_name, 'labels')
    os.makedirs(save_vis_dir, exist_ok=True)
    os.makedirs(save_lbl_dir, exist_ok=True)

    # 3. Get Images
    image_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    print(f"   Processing {len(image_files)} images from {part_name}/{input_folder_name} -> {output_split_name}...")

    success_count = 0
    
    for img_path in image_files:
        try:
            filename = os.path.basename(img_path)
            file_id = os.path.splitext(filename)[0]
            
            # Find Ground Truth
            # Standard naming: GT_IMG_1.mat
            gt_name = "GT_" + file_id + ".mat"
            gt_path = os.path.join(gt_dir, gt_name)
            
            # If standard name fails, try direct name (IMG_1.mat)
            if not os.path.exists(gt_path):
                gt_path = os.path.join(gt_dir, file_id + ".mat")
                
            if not os.path.exists(gt_path):
                continue # Skip if no ground truth (can't train on it)

            # --- CORE PROCESSING ---
            data = loadmat(gt_path)
            try: points = data['image_info'][0][0][0][0][0]
            except: points = data['image_info'][0][0]['location'][0][0]

            # Dynamic Image Size
            img_w = np.max(points[:, 0]) + 50
            img_h = np.max(points[:, 1]) + 50

            # 1. Voronoi Pressure Filter
            vor = Voronoi(points)
            high_risk_points = []
            
            for i, region_idx in enumerate(vor.point_region):
                region = vor.regions[region_idx]
                if -1 not in region and len(region) > 0:
                    area = polygon_area(vor.vertices[region])
                    if 0 < area < CRITICAL_AREA_THRESHOLD:
                        high_risk_points.append(points[i])
            
            high_risk_points = np.array(high_risk_points)
            grid_counts = {}

            # 2. DBSCAN Clustering & Counting
            if len(high_risk_points) > 0:
                clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(high_risk_points)
                labels = clustering.labels_
                
                # Only keep points that belong to a cluster (label != -1)
                clustered_points = high_risk_points[labels != -1]
                
                for px, py in clustered_points:
                    c_idx, r_idx, grid_label = get_grid_coords(px, py, img_w, img_h)
                    if grid_label not in grid_counts:
                        grid_counts[grid_label] = {'count': 0, 'c_idx': c_idx, 'r_idx': r_idx}
                    grid_counts[grid_label]['count'] += 1

            # 3. Visualization Generation
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlim(0, img_w); ax.set_ylim(0, img_h); ax.invert_yaxis()
            
            # Draw Grid Lines
            col_w, row_h = img_w / 10, img_h / 10
            for k in range(11):
                ax.axvline(k * col_w, color='gray', linestyle=':', alpha=0.3)
                ax.axhline(k * row_h, color='gray', linestyle=':', alpha=0.3)
            
            # Draw Axis Labels (A-J, 0-9)
            for i in range(10):
                ax.text(i*col_w + col_w/2, -10, chr(ord('A')+i), color='blue', ha='center', fontweight='bold')
                ax.text(-10, i*row_h + row_h/2, str(i), color='blue', va='center', fontweight='bold')

            # Draw "Safe" Crowd
            ax.scatter(points[:, 0], points[:, 1], s=1, c='lightgray', alpha=0.5)

            # 4. JSON Data Generation
            # We create a flattened array (size 100) representing the 10x10 grid
            # 0 = Safe, 1 = Low, 2 = Med, 3 = High, 4 = Critical
            risk_vector = [0] * 100 
            json_risks = {}
            
            for grid_label, info in grid_counts.items():
                level = get_risk_level(info['count'])
                if level:
                    # Visual: Add colored box
                    color = RISK_COLORS[level]
                    rect = patches.Rectangle((info['c_idx']*col_w, info['r_idx']*row_h), col_w, row_h, 
                                           linewidth=1, edgecolor='black', facecolor=color, alpha=0.6)
                    ax.add_patch(rect)
                    
                    # Visual: Add count number
                    ax.text(info['c_idx']*col_w + col_w/2, info['r_idx']*row_h + row_h/2, str(info['count']),
                           ha='center', va='center', color='white', fontweight='bold', fontsize=8)

                    # Data: Update Vector
                    col_idx = ord(grid_label[0]) - ord('A')
                    row_idx = int(grid_label[1])
                    flat_idx = row_idx * 10 + col_idx
                    
                    score = 1 if level=='LOW' else 2 if level=='MEDIUM' else 3 if level=='HIGH' else 4
                    risk_vector[flat_idx] = score
                    
                    # Data: Readable JSON
                    json_risks[grid_label] = {"level": level, "count": info['count']}

            # Save Visual
            ax.axis('off')
            plt.savefig(os.path.join(save_vis_dir, f"{part_name}_{file_id}.png"), bbox_inches='tight')
            plt.close(fig)

            # Save JSON
            with open(os.path.join(save_lbl_dir, f"{part_name}_{file_id}.json"), 'w') as f:
                json.dump({
                    "image_id": f"{part_name}_{file_id}",
                    "grid_vector": risk_vector,  # Crucial for AI training
                    "readable_risks": json_risks
                }, f)
            
            success_count += 1

        except Exception as e:
            # print(f"Error on {img_path}: {e}") # Uncomment to debug specific files
            pass

    print(f"   -> Successfully saved {success_count} files.")

# --- EXECUTION ---
if __name__ == "__main__":
    if os.path.exists(BASE_PATH):
        print("Starting Data Preparation...")
        
        # 1. Process Training Data (Part A + B) -> 'train'
        process_folder('part_A', 'train_data', 'train')
        process_folder('part_B', 'train_data', 'train')
        
        # 2. Process Validation Data (Part A + B) -> 'val'
        # (This uses the folder you created in the previous step)
        process_folder('part_A', 'validation_data', 'val')
        process_folder('part_B', 'validation_data', 'val')
        
        # 3. Process Test Data (Part A + B) -> 'test'
        process_folder('part_A', 'test_data', 'test')
        process_folder('part_B', 'test_data', 'test')
        
        print(f"\nDONE! Dataset is ready at: {os.path.abspath(OUTPUT_ROOT)}")
    else:
        print(f"Error: Folder '{BASE_PATH}' not found.")