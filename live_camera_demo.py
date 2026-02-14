import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_PATH = 'crowd_risk_model.pth'
CAMERA_SOURCE = 0  # 0 for webcam
RISK_THRESHOLD = 0.3 # Lowered slightly so it catches people faster
OPACITY = 0.6 

# --- SENSITIVITY BOOSTER ---
# 1.0 = Normal (Needs 9+ people for Red)
# 2.0 = High (Needs ~3 people for Red)
# Increase this if you want it MORE sensitive. Decrease if too noisy.
SENSITIVITY = 2.0 

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
    def forward(self, x): return self.backbone(x)

# --- 2. SETUP ---
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found!")
        return None, None
    
    model = CrowdRiskNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# --- 3. MAIN LOOP ---
def run_live_cam():
    model, device = load_model()
    if not model: return

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("--- LIVE CROWD RISK SYSTEM STARTED ---")
    print(f"Sensitivity Multiplier: {SENSITIVITY}x")
    print("Press 'q' to quit.")

    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Resize for Model
        display_frame = cv2.resize(frame, (512, 512))
        pil_img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            # Get raw grid (0.0 to 4.0)
            raw_grid = output.cpu().numpy().flatten().reshape(10, 10)
            
            # --- APPLY SENSITIVITY BOOST ---
            # We multiply by our factor, then clamp to max 4.0
            risk_grid = np.clip(raw_grid * SENSITIVITY, 0, 4)

        # Visualization
        
        # A. Heatmap (Right Side)
        heatmap_overlay = cv2.resize(risk_grid, (512, 512), interpolation=cv2.INTER_CUBIC)
        heatmap_norm = np.uint8(255 * (heatmap_overlay / 4.0))
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # B. Tactical Grid (Left Side)
        grid_overlay = display_frame.copy()
        block_size = 512 // 10
        
        for i in range(10):
            for j in range(10):
                risk_val = risk_grid[i, j]
                
                if risk_val > RISK_THRESHOLD:
                    # STRICTER COLOR THRESHOLDS
                    # Because we boosted the values, we can keep standard thresholds
                    if risk_val < 1.5: color = (0, 255, 0)      # Green (Low density)
                    elif risk_val < 2.5: color = (0, 255, 255)  # Yellow (Medium)
                    elif risk_val < 3.5: color = (0, 165, 255)  # Orange (High)
                    else: color = (0, 0, 255)                   # Red (Critical - 3+ people now hits this)
                    
                    x1, y1 = j * block_size, i * block_size
                    x2, y2 = x1 + block_size, y1 + block_size
                    cv2.rectangle(grid_overlay, (x1, y1), (x2, y2), color, -1)

        tactical_view = cv2.addWeighted(grid_overlay, OPACITY, display_frame, 1 - OPACITY, 0)
        combined_view = np.hstack((tactical_view, heatmap_color))
        
        cv2.imshow('Crowd Risk Monitor', combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_cam()