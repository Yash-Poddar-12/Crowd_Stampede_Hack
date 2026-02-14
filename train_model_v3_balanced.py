import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from PIL import Image

# --- CONFIGURATION ---
DATASET_ROOT = 'final_model_dataset'
BATCH_SIZE = 8 
LEARNING_RATE = 0.0001
EPOCHS = 25  
IMAGE_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# --- 1. DATASET LOADER ---
class CrowdRiskDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.split_dir = os.path.join(root_dir, split)
        self.lbl_dir = os.path.join(self.split_dir, 'labels')
        self.transform = transform
        # Load all JSONs
        self.file_ids = [f.replace('.json', '') for f in os.listdir(self.lbl_dir) if f.endswith('.json')]

    def __len__(self): return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        with open(os.path.join(self.lbl_dir, file_id + '.json'), 'r') as f:
            data = json.load(f)
            risk_vector = torch.tensor(data['grid_vector'], dtype=torch.float32)

        # Try loading visualization; if simpler raw images exist, use those
        img_path = os.path.join(self.split_dir, 'visualizations', file_id + '.png')
        image = Image.open(img_path).convert('RGB')
        
        if self.transform: image = self.transform(image)
        return image, risk_vector

# --- 2. BALANCED LOSS FUNCTION ---
class BalancedRiskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # REDUCED PENALTIES:
        # We lowered Critical from 100.0 to 25.0 to stop the "Paranoia"
        self.weights = torch.tensor([1.0, 5.0, 10.0, 15.0, 25.0]).to(device)

    def forward(self, pred, target):
        squared_diff = (pred - target) ** 2
        target_indices = torch.clamp(torch.round(target), 0, 4).long()
        weight_map = self.weights[target_indices]
        loss = squared_diff * weight_map
        return torch.mean(loss)

# --- 3. MODEL ---
class CrowdRiskNet(nn.Module):
    def __init__(self):
        super(CrowdRiskNet, self).__init__()
        # Use default weights (newer syntax)
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4), # Increased Dropout to prevent memorization
            nn.Linear(512, 100)
        )

    def forward(self, x): return self.backbone(x)

# --- 4. TRAINING LOOP ---
def train_system():
    # STRONG AUGMENTATION: Forces model to look at textures, not locations
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5), # Flip left/right
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Change lighting
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # LOAD EVERYTHING (PART A + PART B are already merged in 'train' folder by prepare_dataset.py)
    # The prepare_dataset.py script we ran earlier combined them into final_model_dataset/train
    print("Loading Combined Dataset (A + B)...")
    train_dataset = CrowdRiskDataset(DATASET_ROOT, 'train', train_transforms)
    val_dataset = CrowdRiskDataset(DATASET_ROOT, 'val', val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training on {len(train_dataset)} images.")

    model = CrowdRiskNet().to(device)
    criterion = BalancedRiskLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n--- STARTING BALANCED TRAINING ({EPOCHS} Epochs) ---")
    
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
    print("\nTraining Complete! New balanced 'crowd_risk_model.pth' saved.")

if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT):
        train_system()
    else:
        print("Dataset not found. Please run prepare_dataset.py first.")