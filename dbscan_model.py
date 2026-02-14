import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Voronoi
from scipy.io import loadmat
from sklearn.cluster import DBSCAN
import os

# --- CONFIGURATION ---
# Use raw string (r'...') to avoid path errors
FILE_PATH = r'ShanghaiTech\part_A\train_data\ground-truth\GT_IMG_1.mat'

# Pressure Threshold (Pixels) - Smaller area = Higher Pressure
CRITICAL_AREA_THRESHOLD = 200  

# Clustering (DBSCAN)
DBSCAN_EPS = 50         # Max distance to be neighbors
DBSCAN_MIN_SAMPLES = 5  # Min people to form a cluster

# --- HELPER FUNCTIONS ---
def polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_grid_label(x, y, img_width, img_height):
    """Converts (x,y) coordinates to Grid Label (e.g., 'C4')."""
    col_width = img_width / 10
    row_height = img_height / 10
    
    col_idx = int(max(0, min(x // col_width, 9)))
    row_idx = int(max(0, min(y // row_height, 9)))
    
    col_label = chr(ord('A') + col_idx)
    return f"{col_label}{row_idx}", col_idx, row_idx

# --- MAIN LOGIC ---
if os.path.exists(FILE_PATH):
    print(f"--- Processing {os.path.basename(FILE_PATH)} ---")
    
    # 1. Load Data
    data = loadmat(FILE_PATH)
    points = data['image_info'][0][0][0][0][0]
    
    # Estimate Image Size
    img_w = np.max(points[:, 0]) + 50
    img_h = np.max(points[:, 1]) + 50
    
    print(f"Loaded {len(points)} people.")
    
    # 2. Identify High-Risk Individuals (Voronoi)
    vor = Voronoi(points)
    high_risk_points = []
    
    # vor.point_region is an array where index 'i' corresponds to 'points[i]'
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        
        # Check if region is valid (not infinite -1, and has vertices)
        if -1 not in region and len(region) > 0:
            area = polygon_area(vor.vertices[region])
            
            # If area is tiny (High Pressure), mark this person
            if 0 < area < CRITICAL_AREA_THRESHOLD:
                high_risk_points.append(points[i])

    high_risk_points = np.array(high_risk_points)
    
    if len(high_risk_points) == 0:
        print("No high-risk zones detected.")
    else:
        print(f"Found {len(high_risk_points)} individuals in high-pressure zones.")

        # 3. CLUSTERING (DBSCAN)
        # fit_predict returns labels directly (-1 is noise)
        labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(high_risk_points)
        unique_labels = set(labels) - {-1}
        
        print(f"Identified {len(unique_labels)} distinct stampede clusters.")
        
        # 4. VISUALIZATION
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw 10x10 Grid
        col_w, row_h = img_w / 10, img_h / 10
        for i in range(11):
            ax.axvline(i * col_w, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(i * row_h, color='gray', linestyle=':', alpha=0.5)
            if i < 10:
                ax.text(i * col_w + 5, 15, chr(ord('A') + i), color='blue', fontsize=8)
                ax.text(5, i * row_h + 15, str(i), color='blue', fontsize=8)

        # Plot Normal Crowd (Faint)
        ax.scatter(points[:, 0], points[:, 1], s=2, c='gray', alpha=0.2, label="Normal")
        
        grid_alerts = []
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            # Extract points for this cluster
            cluster_data = high_risk_points[labels == k]
            
            # Plot Cluster
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=20, color=col, edgecolors='black', label=f"Cluster {k}")
            
            # Bounding Box & Centroid
            centroid = np.mean(cluster_data, axis=0)
            rect_x, rect_y = np.min(cluster_data, axis=0)
            rect_w, rect_h = np.max(cluster_data, axis=0) - [rect_x, rect_y]
            
            ax.add_patch(patches.Rectangle((rect_x, rect_y), rect_w, rect_h, linewidth=2, edgecolor='red', facecolor='none'))

            # Get Grid Sector
            grid_code, c_idx, r_idx = get_grid_label(centroid[0], centroid[1], img_w, img_h)
            grid_alerts.append(grid_code)
            
            # Highlight Grid Sector
            ax.add_patch(patches.Rectangle((c_idx * col_w, r_idx * row_h), col_w, row_h, color='red', alpha=0.15))
            ax.text(centroid[0], centroid[1] - 10, grid_code, color='red', fontweight='bold', ha='center')

        # Final Plot Settings
        ax.set_title(f"RISK MAP: {len(unique_labels)} Clusters Detected\nSectors: {', '.join(grid_alerts)}")
        ax.set_xlim(0, img_w)
        ax.set_ylim(0, img_h)
        ax.invert_yaxis()
        plt.legend()
        plt.show()
        
        print("\n--- SYSTEM ALERT OUTPUT ---")
        print(f"Risk Clusters: {len(unique_labels)}")
        print(f"Alert Grid IDs: {grid_alerts}")

else:
    print(f"Error: File not found at {FILE_PATH}")