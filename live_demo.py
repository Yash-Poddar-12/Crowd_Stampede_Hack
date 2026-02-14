import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_PATH = 'crowd_risk_model.pth'
# REPLACE THIS with the path to an image you want to test!
# You can use one from 'ShanghaiTech/part_A/test_data/images/...' to start.
TEST_IMAGE_PATH = r'ShanghaiTech/part_A/test_data/images/IMG_69.jpg' 

# Colors for the overlay
RISK_COLORS = {
    0: None,            # Safe (No color)
    1: (255, 215, 0),   # Level 1: Gold (Low)
    2: (255, 140, 0),   # Level 2: Orange (Medium)
    3: (255, 69, 0),    # Level 3: Red-Orange (High)
    4: (255, 0, 0)      # Level 4: Red (Critical)
}

# --- 1. RE-DEFINE MODEL (Needed to load weights) ---
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

# --- 2. PREDICTION FUNCTION ---
def predict_risk(image_path):
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

    # Prepare Image
    original_img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        # Round to nearest integer (0-4) and clamp
        prediction = torch.clamp(torch.round(output), 0, 4).cpu().numpy().flatten()

    # --- DRAW RESULTS ---
    print("Generating overlay...")
    result_img = original_img.copy()
    draw = ImageDraw.Draw(result_img, "RGBA")
    
    img_w, img_h = result_img.size
    col_w = img_w / 10
    row_h = img_h / 10
    
    risk_count = 0
    max_risk = 0

    # Reshape flat 100-vector to 10x10 grid
    grid = prediction.reshape(10, 10)

    for r in range(10):
        for c in range(10):
            level = int(grid[r, c])
            
            if level > 0:
                risk_count += 1
                max_risk = max(max_risk, level)
                
                # Get color with transparency
                base_color = RISK_COLORS[level]
                fill_color = base_color + (128,) # Add 50% Alpha
                outline_color = base_color + (255,)
                
                # Coordinates
                x1 = c * col_w
                y1 = r * row_h
                x2 = (c + 1) * col_w
                y2 = (r + 1) * row_h
                
                # Draw Box
                draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=2)
                
                # Draw Number
                # (Simple positioning, might need adjustment for very small images)
                text = str(level)
                draw.text((x1 + 5, y1 + 5), text, fill="white")

    # Show Result
    plt.figure(figsize=(10, 10))
    plt.imshow(result_img)
    plt.axis('off')
    plt.title(f"Crowd Analysis\nMax Risk Level: {max_risk}/4 | Active Sectors: {risk_count}")
    plt.show()
    
    # Save it
    save_path = "demo_result.jpg"
    result_img.save(save_path)
    print(f"Saved result to {save_path}")

if __name__ == "__main__":
    if os.path.exists(TEST_IMAGE_PATH):
        predict_risk(TEST_IMAGE_PATH)
    else:
        print(f"Error: Image not found at {TEST_IMAGE_PATH}")
        print("Please edit the 'TEST_IMAGE_PATH' variable in the script.")