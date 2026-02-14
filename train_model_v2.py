import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# --- CONFIGURATION ---
DATASET_ROOT = 'final_model_dataset'
BATCH_SIZE = 8
LEARNING_RATE = 0.0001  # Lowered LR for stability
EPOCHS = 30             # Increased Epochs
IMAGE_SIZE = 512

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# --- 1. DATASET LOADER ---
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
            # Flatten vector
            risk_vector = torch.tensor(data['grid_vector'], dtype=torch.float32)

        # Load Visualization Image (contains grid context)
        img_path = os.path.join(self.split_dir, 'visualizations', file_id + '.png')
        image = Image.open(img_path).convert('RGB')
        
        if self.transform: image = self.transform(image)
        return image, risk_vector

# --- 2. CUSTOM LOSS FUNCTION (THE FIX) ---
class WeightedRiskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # PENALTY WEIGHTS:
        # Level 0 (Safe): 1x Penalty
        # Level 1 (Low):  10x Penalty
        # Level 2 (Med):  20x Penalty
        # Level 3 (High): 50x Penalty
        # Level 4 (Crit): 100x Penalty
        self.weights = torch.tensor([1.0, 10.0, 20.0, 50.0, 100.0]).to(device)

    def forward(self, pred, target):
        # Calculate standard Squared Error
        squared_diff = (pred - target) ** 2
        
        # Find which weight to apply based on the REAL answer (target)
        # We round the target (0, 1, 2, 3, 4) to use it as an index
        target_indices = torch.clamp(torch.round(target), 0, 4).long()
        
        # Get the weight map for this specific batch
        weight_map = self.weights[target_indices]
        
        # Multiply error by the penalty weight
        loss = squared_diff * weight_map
        
        return torch.mean(loss)

# --- 3. THE MODEL ---
class CrowdRiskNet(nn.Module):
    def __init__(self):
        super(CrowdRiskNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 100) # Output 100 grid scores
        )

    def forward(self, x): return self.backbone(x)

# --- 4. TRAINING LOOP ---
def train_system():
    # Stronger Augmentation to create more "data"
    data_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Random lighting
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CrowdRiskDataset(DATASET_ROOT, 'train', data_transforms)
    val_dataset = CrowdRiskDataset(DATASET_ROOT, 'val', data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = CrowdRiskNet().to(device)
    
    # USE THE NEW LOSS FUNCTION
    criterion = WeightedRiskLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"--- STARTING WEIGHTED TRAINING ({EPOCHS} Epochs) ---")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), "crowd_risk_model.pth")
    print("\nTraining Complete! New 'crowd_risk_model.pth' saved.")

if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT):
        train_system()
    else:
        print("Dataset not found.")