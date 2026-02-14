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
BASE_PATH = 'ShanghaiTech'  # Folder containing part_A and part_B
OUTPUT_ROOT = 'training_data_model_data'

# --- TUNING PARAMETERS ---
# 250 pixels = ~16x16 pixel area.
CRITICAL_AREA_THRESHOLD = 250 

DBSCAN_EPS = 50 
DBSCAN_MIN_SAMPLES = 5

# --- HELPER FUNCTIONS ---
def polygon_area(vertices):
    """Calculates polygon area using Shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_grid_coords(x, y, img_w, img_h):
    """Returns grid indices (col_idx, row_idx) and label (e.g., 'E5')."""
    col_w = img_w / 10
    row_h = img_h / 10
    
    col_idx = int(max(0, min(x // col_w, 9)))
    row_idx = int(max(0, min(y // row_h, 9)))
    
    col_char = chr(ord('A') + col_idx)
    return col_idx, row_idx, f"{col_char}{row_idx}"

def process_dataset_part(part_name):
    print(f"\n================ PROCESSING {part_name} ================")
    
    # 1. Setup Input Paths
    img_dir = os.path.join(BASE_PATH, part_name, 'train_data', 'images')
    gt_dir = os.path.join(BASE_PATH, part_name, 'train_data', 'ground-truth')
    
    # 2. Setup Output Paths
    out_part_dir = os.path.join(OUTPUT_ROOT, part_name)
    out_vis_dir = os.path.join(out_part_dir, 'visualizations') # Saves images
    out_lbl_dir = os.path.join(out_part_dir, 'labels')         # Saves JSON
    
    os.makedirs(out_vis_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    # Get all images
    image_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    total_imgs = len(image_files)
    
    print(f"Found {total_imgs} images in {part_name}. Starting batch job...")

    for idx, img_path in enumerate(image_files):
        # File ID (e.g., IMG_1)
        filename = os.path.basename(img_path)
        file_id = os.path.splitext(filename)[0]
        
        # Construct GT path
        gt_name = "GT_" + file_id + ".mat"
        gt_path = os.path.join(gt_dir, gt_name)
        
        if not os.path.exists(gt_path):
            continue
            
        try:
            # --- CORE LOGIC START ---
            data = loadmat(gt_path)
            
            # Handle Part A vs Part B structure difference
            try:
                points = data['image_info'][0][0][0][0][0]
            except:
                points = data['image_info'][0][0]['location'][0][0]

            # Get dimensions (Dynamic estimation)
            img_w = np.max(points[:, 0]) + 20
            img_h = np.max(points[:, 1]) + 20
            
            # Voronoi Analysis
            vor = Voronoi(points)
            high_risk_points = []
            
            # Collect high-risk points based on density threshold
            for i, region_idx in enumerate(vor.point_region):
                region = vor.regions[region_idx]
                if -1 not in region and len(region) > 0:
                    area = polygon_area(vor.vertices[region])
                    if 0 < area < CRITICAL_AREA_THRESHOLD:
                        high_risk_points.append(points[i])
            
            high_risk_points = np.array(high_risk_points)
            
            # This SET will ensure we save every unique active grid sector
            all_active_grids = set()
            
            # Visualization Setup
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlim(0, img_w)
            ax.set_ylim(0, img_h)
            ax.invert_yaxis()
            
            # Draw 10x10 Grid Lines
            col_w, row_h = img_w / 10, img_h / 10
            for k in range(11):
                ax.axvline(k * col_w, color='gray', linestyle=':', alpha=0.3)
                ax.axhline(k * row_h, color='gray', linestyle=':', alpha=0.3)

            # DBSCAN Clustering
            if len(high_risk_points) > 0:
                clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(high_risk_points)
                labels = clustering.labels_
                unique_labels = set(labels) - {-1}
                
                # Color Map (Blue -> Red based on cluster ID)
                colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_labels)))
                
                for k, col in zip(unique_labels, colors):
                    class_member_mask = (labels == k)
                    cluster_data = high_risk_points[class_member_mask]
                    
                    # 1. Plot the points (dots) for this cluster
                    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=10, color=col)
                    
                    # 2. Iterate through EVERY point to find ALL active grid cells
                    cluster_grids = set()
                    for px, py in cluster_data:
                        c_idx, r_idx, grid_label = get_grid_coords(px, py, img_w, img_h)
                        cluster_grids.add((c_idx, r_idx, grid_label))
                        all_active_grids.add(grid_label)

                    # 3. Highlight ALL grid cells involved with this cluster
                    for c_idx, r_idx, grid_label in cluster_grids:
                        rect = patches.Rectangle((c_idx * col_w, r_idx * row_h), col_w, row_h, 
                                               linewidth=1, edgecolor=col, facecolor=col, alpha=0.3)
                        ax.add_patch(rect)
                        # Label the grid cell
                        ax.text(c_idx * col_w + 10, r_idx * row_h + 20, grid_label, 
                                color='black', fontsize=6, fontweight='bold')

            # Finalize Plot
            # Convert set to sorted list for clean display
            sorted_grids = sorted(list(all_active_grids))
            title_text = f"{file_id} | Active Zones: {len(sorted_grids)}"
            ax.set_title(title_text)
            ax.axis('off')
            
            # Save Image
            save_img_path = os.path.join(out_vis_dir, f"{file_id}_analysis.png")
            plt.savefig(save_img_path, bbox_inches='tight')
            plt.close(fig) 
            
            # Save JSON Data (With complete grid list)
            save_json_path = os.path.join(out_lbl_dir, f"{file_id}_data.json")
            record = {
                "image_id": file_id,
                "risk_count": len(high_risk_points),
                "clusters_detected": len(set(labels) - {-1}) if len(high_risk_points) > 0 else 0,
                "active_grid_sectors": sorted_grids  # SAVES ALL GRIDS (e.g. ["E4", "E5", "E6"])
            }
            with open(save_json_path, 'w') as f:
                json.dump(record, f, indent=4)
                
            if idx % 10 == 0:
                print(f"Processed {idx}/{total_imgs} images...")

        except Exception as e:
            print(f"Error processing {file_id}: {str(e)}")

# --- EXECUTION ---
if __name__ == "__main__":
    if os.path.exists(BASE_PATH):
        # Clears the folder structure or overwrites existing files
        process_dataset_part('part_A')
        process_dataset_part('part_B')
        print(f"\nDONE! All data saved to: {os.path.abspath(OUTPUT_ROOT)}")
    else:
        print(f"Error: Could not find '{BASE_PATH}'. Run this from the simulator folder.")