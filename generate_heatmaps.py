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

# --- CONFIGURATION ---
MODEL_PATH = 'crowd_risk_model.pth'

# Folders to process
INPUT_FOLDERS = [
    r'ShanghaiTech/part_A/test_data/images',
    r'ShanghaiTech/part_B/test_data/images'
]

OUTPUT_FOLDER = 'final_smoothed_results'
IMG_SIZE = 512 # Standard size for processing

# --- 1. MODEL ARCHITECTURE ---
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
    def forward(self, x):
        return self.backbone(x)

# --- 2. LOAD THE BRAIN ---
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model file not found!")
        return None, None

    model = CrowdRiskNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None, None

# --- 3. GENERATE HIGH-RES HEATMAP ---
def process_image(model, device, image_path, dataset_part):
    filename = os.path.basename(image_path)
    
    try:
        # A. Load & Preprocess
        original_img = Image.open(image_path).convert('RGB')
        # Resize original for display consistency
        display_img = original_img.resize((IMG_SIZE, IMG_SIZE))
        
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(original_img).unsqueeze(0).to(device)

        # B. Inference
        with torch.no_grad():
            output = model(input_tensor)
            raw_prediction = torch.clamp(output, 0, 4).cpu().numpy().flatten()

        # C. VISUALIZATION MAGIC (The Fix)
        
        # 1. The Raw 10x10 Grid
        grid_10x10 = raw_prediction.reshape(10, 10)
        
        # 2. The Ultra-Smooth Heatmap
        # Calculate zoom factor to go from 10x10 to 512x512
        zoom_factor = IMG_SIZE / 10 
        # order=3 uses cubic interpolation for smooth curves
        heatmap_smooth = scipy.ndimage.zoom(grid_10x10, zoom_factor, order=3)
        # Clip artifacts that might go slightly below 0 or above 4 due to smoothing math
        heatmap_smooth = np.clip(heatmap_smooth, 0, 4) 

        # D. Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        # LEFT PLOT: Tactical Grid overlay
        ax1.imshow(display_img)
        ax1.set_title(f"[{dataset_part}] Tactical Grid (raw data)\nActive Risk Sectors: {np.sum(grid_10x10 > 0.8)}", fontsize=14, fontweight='bold')
        
        masked_grid = np.ma.masked_where(grid_10x10 < 0.8, grid_10x10)
        # We stretch the 10x10 grid over the 512x512 image using 'nearest' to keep it blocky
        ax1.imshow(masked_grid, cmap='jet', alpha=0.55, vmin=0, vmax=4, 
                   extent=(0, IMG_SIZE, IMG_SIZE, 0), interpolation='nearest')
        ax1.axis('off')

        # RIGHT PLOT: Strategic Heatmap (High Res Smooth)
        # Use interpolation='bicubic' for the final rendering pass
        im2 = ax2.imshow(heatmap_smooth, cmap='jet', vmin=0, vmax=4, interpolation='bicubic')
        ax2.set_title("Strategic Heatmap (High-Def)\nCrowd Pressure Gradient", fontsize=14, fontweight='bold')
        ax2.axis('off')

        # Add Colorbar
        cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Risk Level (0=Safe, 4=Critical)', rotation=270, labelpad=20)

        # E. Save
        save_name = f"HD_Result_{dataset_part}_{filename}"
        save_path = os.path.join(OUTPUT_FOLDER, save_name)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
    except Exception as e:
        print(f"Failed to process {filename}: {e}")

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    model, device = load_model()

    if model:
        print("Model loaded. Starting High-Def processing...")
        total_start = time.time()
        count = 0

        for folder in INPUT_FOLDERS:
            if 'part_A' in folder: part_label = "PartA"
            elif 'part_B' in folder: part_label = "PartB"
            else: part_label = "Unknown"

            print(f"\n--- Processing {part_label} ---")
            
            if os.path.exists(folder):
                types = ('*.jpg', '*.jpeg', '*.png')
                images = []
                for ext in types:
                    images.extend(glob.glob(os.path.join(folder, ext)))
                
                print(f"Found {len(images)} images.")
                for i, img in enumerate(images):
                    process_image(model, device, img, part_label)
                    count += 1
                    if i % 10 == 0: print(f"  Processed {i} images...")
            else:
                print(f"Folder not found: {folder}")

        print(f"\nDONE! Processed {count} images in {time.time() - total_start:.2f} seconds.")
        print(f"Check the new '{OUTPUT_FOLDER}' folder.")