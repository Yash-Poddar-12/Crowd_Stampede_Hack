import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import os
import glob
import time
import math

# --- CONFIGURATION ---
MODEL_PATH = 'crowd_risk_model.pth'
INPUT_FOLDERS = [
    r'ShanghaiTech/part_A/test_data/images',
    r'ShanghaiTech/part_B/test_data/images'
]
OUTPUT_FOLDER = 'final_accurate_results'
TILE_SIZE = 512  # Must match training size

# --- 1. MODEL ---
class CrowdRiskNet(nn.Module):
    def __init__(self):
        super(CrowdRiskNet, self).__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 100)
        )
    def forward(self, x): return self.backbone(x)

# --- 2. LOAD MODEL ---
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH): return None, None
    model = CrowdRiskNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

# --- 3. TILED INFERENCE LOGIC ---
def predict_tile(model, device, tile):
    """Runs prediction on a single 512x512 tile"""
    transform = transforms.Compose([
        transforms.Resize((TILE_SIZE, TILE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(tile).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        # Output is 100 values (10x10 grid for this tile)
        grid = torch.clamp(output, 0, 4).cpu().numpy().flatten().reshape(10, 10)
    return grid

def process_image_tiled(model, device, image_path, dataset_part):
    filename = os.path.basename(image_path)
    try:
        original_img = Image.open(image_path).convert('RGB')
        W, H = original_img.size
        
        # Calculate how many tiles we need
        n_cols = math.ceil(W / TILE_SIZE)
        n_rows = math.ceil(H / TILE_SIZE)
        
        # Create a blank canvas for the full heatmap (High Res)
        # Each tile (512px) outputs a 10x10 grid. 
        # So the "Heatmap Resolution" is (H // 51.2, W // 51.2)
        heatmap_H = n_rows * 10
        heatmap_W = n_cols * 10
        full_heatmap_grid = np.zeros((heatmap_H, heatmap_W))
        
        # --- PROCESS TILES ---
        for row in range(n_rows):
            for col in range(n_cols):
                # Define Crop Box
                x1 = col * TILE_SIZE
                y1 = row * TILE_SIZE
                x2 = min(x1 + TILE_SIZE, W)
                y2 = min(y1 + TILE_SIZE, H)
                
                # Crop and Pad (if edge tile is smaller than 512)
                tile = original_img.crop((x1, y1, x2, y2))
                if tile.size != (TILE_SIZE, TILE_SIZE):
                    new_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE))
                    new_tile.paste(tile, (0, 0))
                    tile = new_tile
                
                # Predict
                tile_grid = predict_tile(model, device, tile)
                
                # Place into full grid
                # Note: We might clip the last row/col if it was padded, 
                # but for visualization, simply placing it is fine.
                full_heatmap_grid[row*10 : (row+1)*10, col*10 : (col+1)*10] = tile_grid

        # --- SMOOTHING ---
        # Resize the blocky grid to match the original image size
        zoom_y = H / heatmap_H
        zoom_x = W / heatmap_W
        heatmap_smooth = scipy.ndimage.zoom(full_heatmap_grid, (zoom_y, zoom_x), order=3)
        heatmap_smooth = np.clip(heatmap_smooth, 0, 4)

        # --- VISUALIZATION ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        # LEFT: Tactical (Overlay)
        ax1.imshow(original_img)
        # Create a mask for the overlay (resize blocky grid to image size first)
        grid_overlay = scipy.ndimage.zoom(full_heatmap_grid, (zoom_y, zoom_x), order=0) # Nearest neighbor
        masked_grid = np.ma.masked_where(grid_overlay < 0.8, grid_overlay) # Hide low risk
        
        ax1.imshow(masked_grid, cmap='jet', alpha=0.55, vmin=0, vmax=4, extent=(0, W, H, 0))
        ax1.set_title(f"[{dataset_part}] Tiled Analysis\n(Accurate Scale)", fontsize=14, fontweight='bold')
        ax1.axis('off')

        # RIGHT: Strategic (Smooth)
        im2 = ax2.imshow(heatmap_smooth, cmap='jet', vmin=0, vmax=4, interpolation='bicubic')
        ax2.set_title("Strategic Heatmap\n(Stitched & Smoothed)", fontsize=14, fontweight='bold')
        ax2.axis('off')

        cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Risk Level', rotation=270, labelpad=20)

        save_path = os.path.join(OUTPUT_FOLDER, f"Accurate_{filename}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    except Exception as e:
        print(f"Error on {filename}: {e}")

# --- MAIN ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    model, device = load_model()
    
    if model:
        print("Starting Tiled Inference (High Accuracy)...")
        for folder in INPUT_FOLDERS:
            if 'part_A' in folder: lbl = "PartA"
            elif 'part_B' in folder: lbl = "PartB"
            else: lbl = "Unknown"
            
            if os.path.exists(folder):
                images = glob.glob(os.path.join(folder, "*.jpg"))
                print(f"Processing {len(images)} images in {lbl}...")
                for img in images:
                    process_image_tiled(model, device, img, lbl)