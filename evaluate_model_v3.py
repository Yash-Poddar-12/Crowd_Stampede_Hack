import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# --- CONFIGURATION ---
DATASET_ROOT = 'final_model_dataset'
MODEL_PATH = 'crowd_risk_model.pth' # The new balanced model
OUTPUT_DIR = 'final_metrics_report'
IMAGE_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. MODEL ---
class CrowdRiskNet(nn.Module):
    def __init__(self):
        super(CrowdRiskNet, self).__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 100)
        )
    def forward(self, x): return self.backbone(x)

# --- 2. DATASET ---
class CrowdRiskDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.split_dir = os.path.join(root_dir, split)
        self.lbl_dir = os.path.join(self.split_dir, 'labels')
        self.transform = transform
        self.file_ids = [f.replace('.json', '') for f in os.listdir(self.lbl_dir) if f.endswith('.json')]

    def __len__(self): return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        with open(os.path.join(self.lbl_dir, file_id + '.json'), 'r') as f:
            data = json.load(f)
            risk_vector = torch.tensor(data['grid_vector'], dtype=torch.float32)
        return risk_vector, file_id # We just need data for metrics

# --- 3. EVALUATION ---
def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading {MODEL_PATH}...")
    model = CrowdRiskNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Check on TEST data (Unseen)
    test_dataset = CrowdRiskDataset(DATASET_ROOT, 'test', transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Evaluating on {len(test_dataset)} test images...")

    total_grids = 0
    correct_grids = 0
    critical_misses = 0
    total_criticals = 0
    
    # Load images on fly to save RAM
    with torch.no_grad():
        for i, (label, file_id) in enumerate(test_loader):
            file_id = file_id[0]
            # Load Image manually
            img_path = os.path.join(DATASET_ROOT, 'test', 'visualizations', file_id + '.png')
            if not os.path.exists(img_path): continue
            
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            label = label.to(device)

            output = model(input_tensor)
            prediction = torch.clamp(torch.round(output), 0, 4)
            
            flat_pred = prediction.cpu().numpy().flatten()
            flat_real = label.cpu().numpy().flatten()
            
            # Accuracy
            total_grids += 100
            correct_grids += np.sum(flat_pred == flat_real)
            
            # Critical Safety Check (Real is 4, Pred is < 3)
            crit_mask = (flat_real == 4)
            total_criticals += np.sum(crit_mask)
            dangerous_misses = np.sum((flat_real == 4) & (flat_pred < 3))
            critical_misses += dangerous_misses

    # Stats
    accuracy = (correct_grids / total_grids) * 100
    safety_score = 100.0
    if total_criticals > 0:
        safety_score = 100 - ((critical_misses / total_criticals) * 100)

    print("\n" + "="*30)
    print("   FINAL METRICS (BALANCED MODEL)   ")
    print("="*30)
    print(f"Overall Accuracy:         {accuracy:.2f}%")
    print(f"Critical Detection Rate:  {safety_score:.2f}%")
    print(f"Total Critical Misses:    {critical_misses} (out of {total_criticals} danger zones)")
    print("="*30)

if __name__ == "__main__":
    evaluate()