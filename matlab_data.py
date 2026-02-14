from scipy.spatial import Voronoi
import numpy as np
from scipy.io import loadmat
import os

# --- HELPER FUNCTION (Required for the code to run) ---
def polygon_area(vertices):
    """Calculates the area of a polygon using the Shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# --- CONFIGURATION ---
CRITICAL_THRESHOLD = 0.05  # Adjust this threshold as needed
# Use r'' to treat backslashes as normal text, not special characters
FILE_PATH = r'ShanghaiTech\part_A\train_data\ground-truth\GT_IMG_1.mat'

# --- MAIN LOGIC ---
if os.path.exists(FILE_PATH):
    # 1. Get head coordinates from ShanghaiTech label file
    data = loadmat(FILE_PATH)

    # 2. Create the Voronoi diagram
    # Note: Adjust array indexing if using Part B structure
    points = data['image_info'][0][0][0][0][0] 
    vor = Voronoi(points)

    # 3. Calculate "Pressure" for each person
    risk_zones = []
    print(f"Analyzing {len(points)} people...")

    for region_index in vor.point_region:
        region = vor.regions[region_index]
        
        # Check for valid region (not infinite, not empty)
        if -1 not in region and len(region) > 0:
            # Calculate area
            area = polygon_area(vor.vertices[region])
            
            # Avoid division by zero for glitchy tiny areas
            if area > 0:
                # INVERSE relationship: Tiny area = HUGE Pressure
                pressure_score = 1.0 / area 
                
                if pressure_score > CRITICAL_THRESHOLD:
                    risk_zones.append("HIGH_RISK")

    print(f"Analysis Done. Found {len(risk_zones)} high-risk zones.")

else:
    print(f"Error: File not found at {FILE_PATH}")
    print("Check if the file was moved to 'validation_data' or if the name is correct.")