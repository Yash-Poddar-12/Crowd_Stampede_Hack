import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# --- CONFIGURATION ---
DATASET_ROOT = 'final_model_dataset'
MODEL_PATH = 'crowd_risk_model.pth'
OUTPUT_DIR = 'evaluation_results'
IMAGE_SIZE = 512
BATCH_SIZE = 1 # Run one by one to visualize easily

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. RE-DEFINE THE MODEL (Required to load weights) ---
class CrowdRiskNet(nn.Module):
    def __init__(self):
        super(CrowdRiskNet, self).__init__()
        self.backbone = models.resnet18(pretrained=False) # Weights loaded later
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 100)
        )
    def forward(self, x):
        return self.backbone(x)

# --- 2. DATASET LOADER (Same as before) ---
class CrowdRiskDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.split_dir = os.path.join(root_dir, split)
        self.lbl_dir = os.path.join(self.split_dir, 'labels')
        self.transform = transform
        self.file_ids = [f.replace('.json', '') for f in os.listdir(self.lbl_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        lbl_path = os.path.join(self.lbl_dir, file_id + '.json')
        with open(lbl_path, 'r') as f:
            data = json.load(f)
            risk_vector = torch.tensor(data['grid_vector'], dtype=torch.float32)
        
        # Load the visualization image for context
        img_path = os.path.join(self.split_dir, 'visualizations', file_id + '.png')
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, risk_vector, file_id

# --- 3. EVALUATION LOGIC ---
def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Model
    print(f"Loading {MODEL_PATH}...")
    model = CrowdRiskNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load Test Data
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = CrowdRiskDataset(DATASET_ROOT, 'test', transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Evaluating on {len(test_dataset)} test images...")

    # Metrics
    total_grids = 0
    correct_grids = 0
    critical_misses = 0 # Predicted Safe when actually Critical
    total_criticals = 0
    
    # Process
    with torch.no_grad():
        for i, (image, label, file_id) in enumerate(test_loader):
            image = image.to(device)
            label = label.to(device)
            
            # Predict
            output = model(image)
            
            # Convert to Integer Levels (Round 2.8 -> 3)
            # We clamp between 0 and 4
            prediction = torch.clamp(torch.round(output), 0, 4)
            
            # Calculate Stats
            flat_pred = prediction.cpu().numpy().flatten()
            flat_real = label.cpu().numpy().flatten()
            
            total_grids += 100
            correct_grids += np.sum(flat_pred == flat_real)
            
            # Critical Safety Check
            # Mask where Real is Critical (Level 4)
            crit_mask = (flat_real == 4)
            total_criticals += np.sum(crit_mask)
            # Count where Real is 4 but Pred is < 3 (Medium or Low)
            dangerous_misses = np.sum((flat_real == 4) & (flat_pred < 3))
            critical_misses += dangerous_misses

            # Visualize the first 10 images
            if i < 10: 
                visualize_comparison(file_id[0], flat_real, flat_pred)

    # Final Report
    accuracy = (correct_grids / total_grids) * 100
    safety_score = 100
    if total_criticals > 0:
        safety_score = 100 - ((critical_misses / total_criticals) * 100)

    print("\n" + "="*30)
    print("   FINAL VALIDATION RESULTS   ")
    print("="*30)
    print(f"Total Grid Cells Checked: {total_grids}")
    print(f"Overall Grid Accuracy:    {accuracy:.2f}%")
    print(f"Safety Score:             {safety_score:.2f}% (Ability to detect Critical zones)")
    print(f"Critical Misses:          {critical_misses} (Times a stampede was missed)")
    print(f"\nVisual proofs saved to folder: '{OUTPUT_DIR}'")

def visualize_comparison(file_id, real, pred):
    """Generates a side-by-side heatmap of Real vs Predicted"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Reshape to 10x10
    real_grid = real.reshape(10, 10)
    pred_grid = pred.reshape(10, 10)
    
    # Plot Real
    ax1.imshow(real_grid, cmap='jet', vmin=0, vmax=4)
    ax1.set_title("Ground Truth (Actual)")
    ax1.axis('off')
    
    # Plot Predicted
    ax2.imshow(pred_grid, cmap='jet', vmin=0, vmax=4)
    ax2.set_title("AI Prediction (Model)")
    ax2.axis('off')
    
    # Add numbers to prediction
    for r in range(10):
        for c in range(10):
            val = int(pred_grid[r, c])
            if val > 0:
                ax2.text(c, r, str(val), ha='center', va='center', color='white', fontweight='bold')

    plt.suptitle(f"Test File: {file_id}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"eval_{file_id}.png"))
    plt.close()

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        evaluate()
    else:
        print("Model file not found!")