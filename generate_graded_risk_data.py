import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Voronoi
from scipy.io import loadmat
from sklearn.cluster import DBSCAN
import os
import json
import glob

# --- CONFIGURATION ---
BASE_PATH = 'ShanghaiTech'
OUTPUT_ROOT = 'training_data_graded_risk'

# --- TUNING PARAMETERS ---
CRITICAL_AREA_THRESHOLD = 250  # Pixels (Personal space limit)
DBSCAN_EPS = 50 
DBSCAN_MIN_SAMPLES = 5

# --- RISK LEVELS (Count of High-Risk People per Grid Cell) ---
# Adjust these based on your camera angle/zoom.
# For ShanghaiTech (dense crowds):
RISK_THRESHOLDS = {
    'LOW': 5,       # 5+ people crushed in one cell = Warning
    'MEDIUM': 15,   # 15+ people = Danger
    'HIGH': 30,     # 30+ people = Severe
    'CRITICAL': 50  # 50+ people = Stampede Likely
}

RISK_COLORS = {
    'LOW': '#FFD700',       # Gold/Yellow
    'MEDIUM': '#FF8C00',    # Dark Orange
    'HIGH': '#FF0000',      # Red
    'CRITICAL': '#8B0000'   # Dark Red
}

# --- HELPER FUNCTIONS ---
def polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_grid_coords(x, y, img_w, img_h):
    """Returns grid indices and label (e.g., 'E5')."""
    col_w = img_w / 10
    row_h = img_h / 10
    
    col_idx = int(max(0, min(x // col_w, 9)))
    row_idx = int(max(0, min(y // row_h, 9)))
    
    col_char = chr(ord('A') + col_idx)
    return col_idx, row_idx, f"{col_char}{row_idx}"

def get_risk_level(count):
    if count >= RISK_THRESHOLDS['CRITICAL']: return 'CRITICAL'
    if count >= RISK_THRESHOLDS['HIGH']: return 'HIGH'
    if count >= RISK_THRESHOLDS['MEDIUM']: return 'MEDIUM'
    if count >= RISK_THRESHOLDS['LOW']: return 'LOW'
    return None

def process_dataset_part(part_name):
    print(f"\n================ PROCESSING {part_name} ================")
    
    img_dir = os.path.join(BASE_PATH, part_name, 'train_data', 'images')
    gt_dir = os.path.join(BASE_PATH, part_name, 'train_data', 'ground-truth')
    
    out_vis_dir = os.path.join(OUTPUT_ROOT, part_name, 'visualizations')
    out_lbl_dir = os.path.join(OUTPUT_ROOT, part_name, 'labels')
    
    os.makedirs(out_vis_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    image_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    
    for idx, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        file_id = os.path.splitext(filename)[0]
        gt_path = os.path.join(gt_dir, "GT_" + file_id + ".mat")
        
        if not os.path.exists(gt_path): continue
            
        try:
            # 1. Load Data
            data = loadmat(gt_path)
            try: points = data['image_info'][0][0][0][0][0]
            except: points = data['image_info'][0][0]['location'][0][0]

            img_w = np.max(points[:, 0]) + 20
            img_h = np.max(points[:, 1]) + 20
            
            # 2. Filter (Voronoi Pressure)
            vor = Voronoi(points)
            high_risk_points = []
            
            for i, region_idx in enumerate(vor.point_region):
                region = vor.regions[region_idx]
                if -1 not in region and len(region) > 0:
                    area = polygon_area(vor.vertices[region])
                    if 0 < area < CRITICAL_AREA_THRESHOLD:
                        high_risk_points.append(points[i])
            
            high_risk_points = np.array(high_risk_points)
            
            # 3. Cluster (DBSCAN) & Aggregation
            grid_counts = {} # Stores count of crushed people per grid cell
            
            if len(high_risk_points) > 0:
                clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(high_risk_points)
                labels = clustering.labels_
                
                # Filter out noise (-1)
                mask = labels != -1
                clustered_points = high_risk_points[mask]
                
                # Count people in each grid cell
                for px, py in clustered_points:
                    c_idx, r_idx, grid_label = get_grid_coords(px, py, img_w, img_h)
                    if grid_label not in grid_counts:
                        grid_counts[grid_label] = {'count': 0, 'c_idx': c_idx, 'r_idx': r_idx}
                    grid_counts[grid_label]['count'] += 1

            # 4. Visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Draw Grid Lines & Custom Axis Labels
            col_w, row_h = img_w / 10, img_h / 10
            
            # Set Ticks exactly at the center of each cell
            ax.set_xticks(np.arange(col_w/2, img_w, col_w))
            ax.set_xticklabels([chr(ord('A')+i) for i in range(10)], fontsize=12, fontweight='bold', color='blue')
            
            ax.set_yticks(np.arange(row_h/2, img_h, row_h))
            ax.set_yticklabels([str(i) for i in range(10)], fontsize=12, fontweight='bold', color='blue')
            
            # Draw actual grid lines
            for k in range(11):
                ax.axvline(k * col_w, color='gray', linestyle=':', alpha=0.5)
                ax.axhline(k * row_h, color='gray', linestyle=':', alpha=0.5)

            # Plot background dots (Safe people)
            ax.scatter(points[:, 0], points[:, 1], s=1, c='lightgray', alpha=0.5)
            
            # 5. Paint the Risk Zones
            json_risk_data = {}
            
            for grid_label, info in grid_counts.items():
                count = info['count']
                level = get_risk_level(count)
                
                if level:
                    # Save to JSON structure
                    json_risk_data[grid_label] = {
                        "level": level,
                        "crushed_person_count": count
                    }
                    
                    # Draw Rectangle on Map
                    c_idx, r_idx = info['c_idx'], info['r_idx']
                    color = RISK_COLORS[level]
                    
                    # Rectangle
                    rect = patches.Rectangle((c_idx * col_w, r_idx * row_h), col_w, row_h, 
                                           linewidth=1, edgecolor='black', facecolor=color, alpha=0.6)
                    ax.add_patch(rect)
                    
                    # Text Count
                    ax.text(c_idx * col_w + col_w/2, r_idx * row_h + row_h/2, str(count),
                           ha='center', va='center', color='white', fontweight='bold')

            # Final Plot Settings
            ax.set_xlim(0, img_w)
            ax.set_ylim(0, img_h)
            ax.invert_yaxis()
            ax.set_title(f"{file_id} Risk Map\nMax Density: {max([x['count'] for x in grid_counts.values()] + [0])} people/cell")
            
            # Save Visual
            plt.savefig(os.path.join(out_vis_dir, f"{file_id}_risk_map.png"), bbox_inches='tight')
            plt.close(fig)
            
            # Save JSON
            with open(os.path.join(out_lbl_dir, f"{file_id}_risk.json"), 'w') as f:
                json.dump({"image_id": file_id, "risks": json_risk_data}, f, indent=4)
                
            if idx % 10 == 0: print(f"Processed {idx} images...")
            
        except Exception as e:
            print(f"Skipping {file_id}: {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    if os.path.exists(BASE_PATH):
        process_dataset_part('part_A')
        process_dataset_part('part_B')
        print(f"\nDONE! Check folder: {OUTPUT_ROOT}")
    else:
        print("Error: ShanghaiTech folder not found.")