import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN
from PIL import Image
from ultralytics import YOLO
import os
import glob
import time

# --- CONFIGURATION ---
INPUT_FOLDERS = [
    r'ShanghaiTech/part_A/test_data/images',
    r'ShanghaiTech/part_B/test_data/images' 
]

OUTPUT_ROOT = 'yolo_batch_results'

# PHYSICS SETTINGS
CRITICAL_SPACE_THRESHOLD = 500  
DBSCAN_EPS = 100
DBSCAN_MIN_SAMPLES = 3

# --- 1. DETECTOR ---
def load_yolo():
    print("Loading YOLOv8x (Extra Large) Model...")
    return YOLO('yolov8x.pt')

def detect_people(model, image_path):
    # FORCE CPU INFERENCE (Fixes the CUDA error)
    results = model.predict(image_path, conf=0.25, classes=0, verbose=False, device='cpu')
    
    # Extract points
    points = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        points.append([center_x, center_y])
        
    return np.array(points), results[0].orig_shape

# --- 2. MATH ---
def polygon_area(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def analyze_risk(points, img_shape):
    H, W = img_shape
    if len(points) < 4: return np.zeros((10, 10)), np.array([])

    dummy_points = [[0,0], [0,H], [W,0], [W,H]]
    all_points = np.vstack([points, dummy_points])
    vor = Voronoi(all_points)
    high_risk_points = []
    
    for i, region_idx in enumerate(vor.point_region):
        if i >= len(points): break
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            area = polygon_area(vor.vertices[region])
            if 0 < area < CRITICAL_SPACE_THRESHOLD:
                high_risk_points.append(points[i])
    
    high_risk_points = np.array(high_risk_points)
    
    # Grid Generation
    risk_grid = np.zeros((10, 10))
    col_w, row_h = W / 10, H / 10

    if len(high_risk_points) > 0:
        clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(high_risk_points)
        labels = clustering.labels_
        clustered_points = high_risk_points[labels != -1]
        
        for px, py in clustered_points:
            c_idx = int(min(px // col_w, 9))
            r_idx = int(min(py // row_h, 9))
            risk_grid[r_idx, c_idx] += 1

    return np.clip(risk_grid / 2.0, 0, 4), high_risk_points

# --- 3. VISUALIZATION ---
def save_report(image_path, risk_grid, all_points, risk_points, output_dir):
    try:
        original_img = Image.open(image_path).convert('RGB')
        W, H = original_img.size
        
        heatmap_40x40 = scipy.ndimage.zoom(risk_grid, 4, order=3)
        heatmap_40x40 = np.clip(heatmap_40x40, 0, 4)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        # TACTICAL
        ax1.imshow(original_img)
        ax1.set_title(f"YOLO Detection\nPeople: {len(all_points)} | Crushed: {len(risk_points)}", fontsize=14, fontweight='bold')
        if len(all_points) > 0:
            ax1.scatter(all_points[:, 0], all_points[:, 1], c='#00FF00', s=10, alpha=0.5)
        if len(risk_points) > 0:
            ax1.scatter(risk_points[:, 0], risk_points[:, 1], c='red', s=30, marker='x')
        
        masked_grid = np.ma.masked_where(risk_grid < 0.5, risk_grid)
        ax1.imshow(masked_grid, cmap='jet', alpha=0.4, vmin=0, vmax=4, extent=(0, W, H, 0))
        ax1.axis('off')

        # STRATEGIC
        im2 = ax2.imshow(heatmap_40x40, cmap='jet', vmin=0, vmax=4)
        ax2.set_title("Density Heatmap (Physics-Based)", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        filename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f"Result_{filename}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig) 
        
    except Exception as e:
        print(f"Skipping visualization for {image_path}: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)
    
    # Initialize Model
    model = load_yolo()
    total_start = time.time()
    count = 0

    for folder in INPUT_FOLDERS:
        if not os.path.exists(folder):
            print(f"Skipping missing folder: {folder}")
            continue

        folder_name = "Part_A" if "part_A" in folder else "Part_B"
        save_dir = os.path.join(OUTPUT_ROOT, folder_name)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        print(f"\n--- Processing {folder_name} ---")
        images = glob.glob(os.path.join(folder, "*.jpg"))
        
        for i, img_path in enumerate(images):
            points, shape = detect_people(model, img_path)
            risk_grid, risk_points = analyze_risk(points, shape)
            save_report(img_path, risk_grid, points, risk_points, save_dir)
            
            print(f"[{i+1}/{len(images)}] Processed {os.path.basename(img_path)}")
            count += 1

    print(f"\nDONE! Processed {count} images in {time.time() - total_start:.2f} seconds.")
    print(f"Results saved in: {OUTPUT_ROOT}")