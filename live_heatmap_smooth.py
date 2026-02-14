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
# Input: Use a folder to process many images at once
INPUT_PATH = r'ShanghaiTech/part_A/test_data/images' 
OUTPUT_FOLDER = 'heatmap_results_smooth'

# Sliding Window Settings
WINDOW_SIZE = 512
STRIDE = 256  # 50% Overlap

# --- MODEL ---
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

def process_sliding_window(model, device, image_path):
    filename = os.path.basename(image_path)
    print(f"Processing: {filename}...")

    try:
        original_img = Image.open(image_path).convert('RGB')
        W, H = original_img.size
        
        heatmap_accumulator = np.zeros((H, W), dtype=np.float32)
        count_accumulator = np.zeros((H, W), dtype=np.float32)

        transform = transforms.Compose([
            transforms.Resize((WINDOW_SIZE, WINDOW_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Slide Window
        for y in range(0, H, STRIDE):
            for x in range(0, W, STRIDE):
                x1, y1 = x, y
                x2, y2 = x + WINDOW_SIZE, y + WINDOW_SIZE
                if x2 > W: x2 = W; x1 = max(0, W - WINDOW_SIZE)
                if y2 > H: y2 = H; y1 = max(0, H - WINDOW_SIZE)
                
                patch = original_img.crop((x1, y1, x2, y2))
                input_tensor = transform(patch).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_grid = torch.clamp(output, 0, 4).cpu().numpy().flatten().reshape(10, 10)
                
                # Expand 10x10 -> Window Size
                pred_expanded = scipy.ndimage.zoom(pred_grid, (y2-y1)/10, order=1)
                # Resize to exact fit if zoom was slightly off
                pred_expanded = np.array(Image.fromarray(pred_expanded).resize((x2-x1, y2-y1), Image.BILINEAR))
                
                heatmap_accumulator[y1:y2, x1:x2] += pred_expanded
                count_accumulator[y1:y2, x1:x2] += 1.0

        final_heatmap = heatmap_accumulator / count_accumulator
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        ax1.imshow(original_img)
        ax1.set_title(f"Tactical View", fontsize=14, fontweight='bold')
        masked_map = np.ma.masked_where(final_heatmap < 0.8, final_heatmap)
        ax1.imshow(masked_map, cmap='jet', alpha=0.55, vmin=0, vmax=4)
        ax1.axis('off')

        im2 = ax2.imshow(final_heatmap, cmap='jet', vmin=0, vmax=4)
        ax2.set_title("Strategic Risk Heatmap (Bias Removed)", fontsize=14, fontweight='bold')
        ax2.axis('off')

        cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Risk Level', rotation=270, labelpad=20)

        save_path = os.path.join(OUTPUT_FOLDER, f"Smooth_{filename}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrowdRiskNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        if os.path.isdir(INPUT_PATH):
            images = glob.glob(os.path.join(INPUT_PATH, "*.jpg"))
            for img in images: process_sliding_window(model, device, img)
        else:
            print("Check your INPUT_PATH")