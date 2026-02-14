import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import os

# --- CONFIGURATION ---
MODEL_PATH = 'crowd_risk_model.pth'
# REPLACE with the image you want to test
TEST_IMAGE_PATH = r'ShanghaiTech/part_A/test_data/images/IMG_10.jpg' 
OUTPUT_FOLDER = 'heatmap_results'

# --- 1. RE-DEFINE MODEL ---
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

# --- 2. MAIN PROCESSING ---
def generate_heatmap_report(image_path):
    # Setup Output Folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = CrowdRiskNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()

    # Load & Transform Image
    original_img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        # Clamp between 0 (Safe) and 4 (Critical)
        raw_prediction = torch.clamp(output, 0, 4).cpu().numpy().flatten()

    # --- 3. DATA PROCESSING ---
    
    # A. The Tactical Grid (10x10)
    grid_10x10 = raw_prediction.reshape(10, 10)
    
    # B. The Strategic Heatmap (40x40)
    # Zoom factor = 4x (10 -> 40). Order=3 gives smooth cubic interpolation.
    heatmap_40x40 = scipy.ndimage.zoom(grid_10x10, 4, order=3)
    heatmap_40x40 = np.clip(heatmap_40x40, 0, 4) # Clean up artifacts

    # --- 4. VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # LEFT: Tactical View (Overlay)
    ax1.imshow(original_img)
    ax1.set_title(f"Tactical Grid (10x10)\nActive Sectors: {np.sum(grid_10x10 > 0.5)}", fontsize=14, fontweight='bold')
    
    # Create a transparency mask (Hide "Safe" blue areas so we see the photo)
    # Pixels < 0.5 (Safe) will be fully transparent
    masked_grid = np.ma.masked_where(grid_10x10 < 0.5, grid_10x10)
    
    # Overlay with 'jet' colormap (Blue=Low -> Red=Critical)
    ax1.imshow(masked_grid, cmap='jet', alpha=0.55, vmin=0, vmax=4, 
               extent=(0, original_img.width, original_img.height, 0))
    ax1.axis('off')

    # RIGHT: Strategic View (Smooth Heatmap)
    im2 = ax2.imshow(heatmap_40x40, cmap='jet', vmin=0, vmax=4)
    ax2.set_title("Strategic Heatmap (40x40)\nCrowd Pressure Gradient", fontsize=14, fontweight='bold')
    
    # Grid Styling
    ax2.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.1)
    ax2.set_xlabel("Sector Width")
    ax2.set_ylabel("Sector Depth")

    # Colorbar
    cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Risk Level (0=Safe, 4=Critical)', rotation=270, labelpad=20)

    # --- 5. SAVE RESULT ---
    filename = os.path.basename(image_path)
    save_name = f"Result_{filename}"
    save_path = os.path.join(OUTPUT_FOLDER, save_name)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
    print(f"SUCCESS: Result saved to: {save_path}")

if __name__ == "__main__":
    if os.path.exists(TEST_IMAGE_PATH):
        generate_heatmap_report(TEST_IMAGE_PATH)
    else:
        print(f"Error: Could not find image at {TEST_IMAGE_PATH}")