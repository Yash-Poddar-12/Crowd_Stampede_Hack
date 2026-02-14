import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.io import loadmat
import os
import glob

# --- CONFIGURATION ---
# Base path to your data
# We use os.path.join for safety, but r"string" also works
BASE_DIR = os.path.join('ShanghaiTech', 'part_A', 'train_data', 'ground-truth')

# Adjust this based on your image resolution. 
CRITICAL_AREA_THRESHOLD = 2000.0 

def polygon_area(vertices):
    """ Calculates the area of a polygon using the Shoelace formula. """
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_first_available_mat_file(directory):
    """ Finds the first .mat file in the directory to ensure code runs. """
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return None

    # Search for any .mat file
    search_path = os.path.join(directory, "*.mat")
    files = glob.glob(search_path)
    
    if files:
        print(f"Found {len(files)} ground-truth files.")
        return files[0] # Return the first one found
    else:
        print("No .mat files found in this directory.")
        return None

def get_crowd_data(file_path):
    """ Loads data from .mat file. """
    print(f"Loading data from: {file_path}")
    
    try:
        data = loadmat(file_path)
        # ShanghaiTech Part A standard structure
        # image_info -> location -> [0][0][0][0][0]
        points = data['image_info'][0][0][0][0][0] 
        return points
    except Exception as e:
        print(f"Error reading .mat structure: {e}")
        # Fallback for some Part B files which are structured differently
        try:
            print("Trying alternate Part B structure...")
            points = data['image_info'][0][0]['location'][0][0]
            return points
        except:
            print("Could not parse this .mat file. Generating MOCK data.")
            return np.random.rand(50, 2) * [800, 600]

def analyze_crowd_pressure():
    # 1. Find a valid file automatically
    mat_file = get_first_available_mat_file(BASE_DIR)
    
    if mat_file:
        points = get_crowd_data(mat_file)
    else:
        print("Using MOCK data (Path issue)...")
        points = np.random.rand(50, 2) * [800, 600]

    # 2. Create Voronoi Diagram
    vor = Voronoi(points)

    # 3. Setup Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the base points (heads)
    ax.plot(points[:, 0], points[:, 1], 'o', color='black', markersize=3, label='Heads')

    risk_count = 0
    total_count = 0

    # 4. Analyze each region (person)
    for region_index in vor.point_region:
        region = vor.regions[region_index]
        
        # Skip invalid/infinite regions
        if not region or -1 in region:
            continue
        
        # Get vertices
        polygon_vertices = vor.vertices[region]
        
        # Calculate Area (Personal Space)
        area = polygon_area(polygon_vertices)
        
        # Filter outliers (infinity edges or glitches)
        if area < 1 or area > 100000: 
            continue

        total_count += 1
        
        # --- PRESSURE LOGIC ---
        if area < CRITICAL_AREA_THRESHOLD:
            color = 'red'     # High Pressure
            risk_count += 1
            alpha = 0.6
        else:
            color = '#90EE90' # Safe (Green)
            alpha = 0.3

        # Draw the polygon
        ax.fill(polygon_vertices[:, 0], polygon_vertices[:, 1], color=color, alpha=alpha, edgecolor='white')

    # 5. Final Output
    title = f"Crowd Pressure Map\nHigh Risk Areas (Red): {risk_count}/{total_count} People detected"
    ax.set_title(title)
    
    # Auto-scale to fit the crowd points
    if len(points) > 0:
        ax.set_xlim(0, np.max(points[:,0]) + 50) 
        ax.set_ylim(0, np.max(points[:,1]) + 50)
    
    ax.invert_yaxis() # Fix upside-down image coordinates
    plt.legend()
    print(f"Analysis Complete. Found {risk_count} people at high risk.")
    plt.show()

if __name__ == "__main__":
    analyze_crowd_pressure()