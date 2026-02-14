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

# LIST of all folders you want to process
INPUT_FOLDERS = [
    r'ShanghaiTech/part_A/test_data/images',
    r'ShanghaiTech/part_B/test_data/images'
]

OUTPUT_FOLDER = 'heatmap_results_combined'

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

# --- 3. PROCESS IMAGE ---
def process_image(model, device, image_path, dataset_part):
    filename = os.path.basename(image_path)
    
    try:
        # Load & Transform
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
            raw_prediction = torch.clamp(output, 0, 4).cpu().numpy().flatten()

        # Data Processing
        grid_10x10 = raw_prediction.reshape(10, 10)
        heatmap_40x40 = scipy.ndimage.zoom(grid_10x10, 4, order=3)
        heatmap_40x40 = np.clip(heatmap_40x40, 0, 4) 

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        # Left: Tactical
        ax1.imshow(original_img)
        ax1.set_title(f"[{dataset_part}] Tactical Grid\nActive Sectors: {np.sum(grid_10x10 > 0.5)}", fontsize=14, fontweight='bold')
        masked_grid = np.ma.masked_where(grid_10x10 < 0.5, grid_10x10)
        ax1.imshow(masked_grid, cmap='jet', alpha=0.55, vmin=0, vmax=4, 
                   extent=(0, original_img.width, original_img.height, 0))
        ax1.axis('off')

        # Right: Strategic
        im2 = ax2.imshow(heatmap_40x40, cmap='jet', vmin=0, vmax=4)
        ax2.set_title(f"Strategic Heatmap\nCrowd Pressure Gradient", fontsize=14, fontweight='bold')
        ax2.axis('off')

        # Colorbar
        cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Risk Level', rotation=270, labelpad=20)

        # Save with Unique Name (e.g., Result_PartA_IMG_1.jpg)
        save_name = f"Result_{dataset_part}_{filename}"
        save_path = os.path.join(OUTPUT_FOLDER, save_name)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
        print(f"   -> Saved: {save_name}")

    except Exception as e:
        print(f"   -> Error on {filename}: {e}")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    model, device = load_model()

    if model:
        for folder in INPUT_FOLDERS:
            # Determine if this is Part A or Part B for labeling
            if 'part_A' in folder:
                part_label = "PartA"
            elif 'part_B' in folder:
                part_label = "PartB"
            else:
                part_label = "Unknown"

            print(f"\n--- Processing Folder: {part_label} ---")
            
            # Find images
            if os.path.exists(folder):
                types = ('*.jpg', '*.jpeg', '*.png')
                images = []
                for ext in types:
                    images.extend(glob.glob(os.path.join(folder, ext)))
                
                print(f"Found {len(images)} images.")
                
                # Process each
                for img in images:
                    process_image(model, device, img, part_label)
            else:
                print(f"Folder not found: {folder}")