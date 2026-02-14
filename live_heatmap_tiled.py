import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import os
import glob

# --- CONFIGURATION ---
MODEL_PATH = 'crowd_risk_model.pth'
# Input Folder or File
INPUT_PATH = r'ShanghaiTech/part_A/test_data/images' 
OUTPUT_FOLDER = 'heatmap_results_tiled'

# --- 1. MODEL DEFINITION ---
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

# --- 2. LOAD MODEL ---
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {MODEL_PATH}...")
    model = CrowdRiskNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# --- 3. TILED PROCESSING LOGIC ---
def predict_patch(model, device, patch_img):
    """Runs inference on a single crop/patch"""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(patch_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        # Clamp 0-4
        pred = torch.clamp(output, 0, 4).cpu().numpy().flatten()
        
    # Reshape this patch's prediction to 10x10
    return pred.reshape(10, 10)

def process_tiled_image(model, device, image_path):
    filename = os.path.basename(image_path)
    print(f"Processing (Tiled): {filename}...")
    
    try:
        original_img = Image.open(image_path).convert('RGB')
        W, H = original_img.size
        
        # 1. SPLIT IMAGE INTO 4 QUADRANTS (2x2 Grid)
        # We ensure they are roughly square for the 512x512 ResNet
        # Top-Left, Top-Right, Bottom-Left, Bottom-Right
        
        # Define crop box: (left, upper, right, lower)
        half_w, half_h = W // 2, H // 2
        
        crops = [
            original_img.crop((0, 0, half_w, half_h)),        # Top-Left
            original_img.crop((half_w, 0, W, half_h)),        # Top-Right
            original_img.crop((0, half_h, half_w, H)),        # Bottom-Left
            original_img.crop((half_w, half_h, W, H))         # Bottom-Right
        ]
        
        # 2. PREDICT ON EACH PATCH
        grids = []
        for patch in crops:
            grids.append(predict_patch(model, device, patch))
            
        # 3. STITCH 4x (10x10) GRIDS INTO 1x (20x20) GRID
        # Top Row: [Top-Left, Top-Right]
        top_row = np.hstack((grids[0], grids[1])) 
        # Bottom Row: [Bottom-Left, Bottom-Right]
        bottom_row = np.hstack((grids[2], grids[3]))
        
        # Combine Rows
        final_grid_20x20 = np.vstack((top_row, bottom_row))
        
        # 4. GENERATE HEATMAP (Interpolate 20x20 -> 80x80)
        # We use a smaller zoom factor because our base grid is higher res now
        heatmap_HD = scipy.ndimage.zoom(final_grid_20x20, 4, order=3)
        heatmap_HD = np.clip(heatmap_HD, 0, 4)

        # --- VISUALIZATION ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        # LEFT: Tactical Grid (Now 20x20 resolution!)
        ax1.imshow(original_img)
        ax1.set_title(f"Tactical Grid (High Res 20x20)\nActive Sectors: {np.sum(final_grid_20x20 > 0.5)}", fontsize=14, fontweight='bold')
        
        masked_grid = np.ma.masked_where(final_grid_20x20 < 0.5, final_grid_20x20)
        ax1.imshow(masked_grid, cmap='jet', alpha=0.55, vmin=0, vmax=4, 
                   extent=(0, W, H, 0))
        ax1.axis('off')

        # RIGHT: Strategic Heatmap
        im2 = ax2.imshow(heatmap_HD, cmap='jet', vmin=0, vmax=4)
        ax2.set_title("Strategic Heatmap (HD)\nCrowd Pressure Gradient", fontsize=14, fontweight='bold')
        ax2.axis('off')

        cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Risk Level', rotation=270, labelpad=20)

        # SAVE
        save_name = f"TiledResult_{filename}"
        save_path = os.path.join(OUTPUT_FOLDER, save_name)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
        print(f"   -> Saved: {save_path}")

    except Exception as e:
        print(f"   -> Error: {e}")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    model, device = load_model()

    if model:
        if os.path.isdir(INPUT_PATH):
            types = ('*.jpg', '*.jpeg', '*.png')
            images = []
            for ext in types:
                images.extend(glob.glob(os.path.join(INPUT_PATH, ext)))
            
            print(f"Processing {len(images)} images in batch...")
            for img in images:
                process_tiled_image(model, device, img)
        elif os.path.isfile(INPUT_PATH):
            process_tiled_image(model, device, INPUT_PATH)
        else:
            print("Path not found.")