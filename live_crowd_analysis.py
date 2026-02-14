import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage
from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN
from PIL import Image
from ultralytics import YOLO
import os

# --- CONFIGURATION ---
# REPLACE THIS with the path to the image you want to test
INPUT_IMAGE_PATH = r'ShanghaiTech/part_A/test_data/images/IMG_10.jpg' 
OUTPUT_FOLDER = 'yolo_density_results'

# --- TUNING THE PHYSICS ---
# How much space (in pixels) does a person need to be "Safe"?
# If their Voronoi region is smaller than this, they are "Crushed".
CRITICAL_SPACE_THRESHOLD = 500  

# Clustering Settings
DBSCAN_EPS = 100        # Max distance to group crushed people together
DBSCAN_MIN_SAMPLES = 3  # Min people to call it a "Cluster"

# --- 1. DETECTOR (YOLOv8) ---
def detect_people(image_path):
    print(f"--- Loading YOLOv8 Model ---")
    # 'yolov8x.pt' is the largest, most accurate model. It downloads automatically.
    model = YOLO('yolov8x.pt') 
    
    print(f"Detecting people in {os.path.basename(image_path)}...")
    # classes=0 ensures we ONLY detect "Person" and ignore cars/bags
    results = model.predict(image_path, conf=0.25, classes=0, verbose=False)
    
    # Extract Center Coordinates (x, y)
    points = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        points.append([center_x, center_y])
        
    return np.array(points), results[0].orig_shape # Returns points and Image (H, W)

# --- 2. THE MATH (Voronoi + Density) ---
def polygon_area(vertices):
    # Shoelace formula to calculate area of a polygon
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def analyze_risk(points, img_shape):
    H, W = img_shape
    
    # Voronoi fails if < 4 points, so we handle empty scenes
    if len(points) < 4:
        return np.zeros((10, 10)), np.array([])

    # Add "Dummy Points" at corners so the math works for edge people
    dummy_points = [[0,0], [0,H], [W,0], [W,H]]
    all_points = np.vstack([points, dummy_points])
    
    vor = Voronoi(all_points)
    high_risk_points = []
    
    # Check every person's region
    for i, region_idx in enumerate(vor.point_region):
        if i >= len(points): break # Stop before dummy points
        
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            area = polygon_area(vor.vertices[region])
            
            # THE CORE LOGIC: Small Area = High Density = High Risk
            if 0 < area < CRITICAL_SPACE_THRESHOLD:
                high_risk_points.append(points[i])
    
    high_risk_points = np.array(high_risk_points)
    
    # Generate 10x10 Grid Data from High Risk Points
    risk_grid = np.zeros((10, 10))
    col_w, row_h = W / 10, H / 10

    if len(high_risk_points) > 0:
        # Optional: Cluster them to find "Stampede Epicenters"
        clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(high_risk_points)
        labels = clustering.labels_
        clustered_points = high_risk_points[labels != -1]
        
        for px, py in clustered_points:
            c_idx = int(min(px // col_w, 9))
            r_idx = int(min(py // row_h, 9))
            risk_grid[r_idx, c_idx] += 1

    # Normalize: If a grid cell has >8 crushed people, it is Level 4 (Critical)
    risk_grid = np.clip(risk_grid / 2.0, 0, 4) 
    
    return risk_grid, high_risk_points

# --- 3. VISUALIZATION ---
def generate_report(image_path, risk_grid, all_points, risk_points):
    original_img = Image.open(image_path).convert('RGB')
    W, H = original_img.size
    
    # Create Smooth Heatmap (40x40) from the 10x10 Grid
    heatmap_40x40 = scipy.ndimage.zoom(risk_grid, 4, order=3)
    heatmap_40x40 = np.clip(heatmap_40x40, 0, 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # --- LEFT: TACTICAL VIEW (Real-World Data) ---
    ax1.imshow(original_img)
    ax1.set_title(f"Tactical Detection (YOLOv8)\nTotal People: {len(all_points)} | Crushed: {len(risk_points)}", fontsize=14, fontweight='bold')
    
    # Draw ALL people (Green Dots)
    if len(all_points) > 0:
        ax1.scatter(all_points[:, 0], all_points[:, 1], c='#00FF00', s=15, alpha=0.6, label='Safe Space')
    
    # Draw CRUSHED people (Red X's)
    if len(risk_points) > 0:
        ax1.scatter(risk_points[:, 0], risk_points[:, 1], c='red', s=40, marker='x', linewidth=2, label='High Density Risk')

    # Overlay the Risk Grid
    masked_grid = np.ma.masked_where(risk_grid < 0.5, risk_grid)
    ax1.imshow(masked_grid, cmap='jet', alpha=0.4, vmin=0, vmax=4, extent=(0, W, H, 0))
    ax1.legend(loc='upper right', frameon=True, facecolor='black', labelcolor='white')
    ax1.axis('off')

    # --- RIGHT: STRATEGIC HEATMAP (Gradient) ---
    im2 = ax2.imshow(heatmap_40x40, cmap='jet', vmin=0, vmax=4)
    ax2.set_title("Strategic Density Heatmap\n(Generated from Coordinates)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Stampede Risk Level', rotation=270, labelpad=20)

    # Save
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    filename = os.path.basename(image_path)
    save_path = os.path.join(OUTPUT_FOLDER, f"Analysis_{filename}")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"SUCCESS: Report saved to {save_path}")

# --- EXECUTION ---
if __name__ == "__main__":
    if os.path.exists(INPUT_IMAGE_PATH):
        # 1. AI Detection
        points, shape = detect_people(INPUT_IMAGE_PATH)
        
        # 2. Mathematical Analysis
        risk_grid, risk_points = analyze_risk(points, shape)
        
        # 3. Generate Report
        generate_report(INPUT_IMAGE_PATH, risk_grid, points, risk_points)
    else:
        print(f"Error: Image not found at {INPUT_IMAGE_PATH}")