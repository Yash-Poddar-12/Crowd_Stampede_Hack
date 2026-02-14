import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import random

# --- CONFIGURATION ---
DATASET_ROOT = 'final_model_dataset'
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
EPOCHS = 30 # Give it enough time to un-learn the bad habits
IMAGE_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# --- 1. SMART DATASET (With Geometric Flipping) ---
class CrowdRiskDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, augment=False):
        self.split_dir = os.path.join(root_dir, split)
        self.lbl_dir = os.path.join(self.split_dir, 'labels')
        self.transform = transform
        self.augment = augment # Only augment training data
        self.file_ids = [f.replace('.json', '') for f in os.listdir(self.lbl_dir) if f.endswith('.json')]

    def __len__(self): return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        
        # Load Labels
        with open(os.path.join(self.lbl_dir, file_id + '.json'), 'r') as f:
            data = json.load(f)
            # Load as numpy for easier manipulation
            risk_vector = np.array(data['grid_vector'], dtype=np.float32)

        # Load Image
        img_path = os.path.join(self.split_dir, 'visualizations', file_id + '.png')
        image = Image.open(img_path).convert('RGB')
        
        # --- SMART AUGMENTATION ---
        if self.augment:
            # 50% chance to flip VERTICALLY (The Anti-Bias Fix)
            if random.random() > 0.5:
                # 1. Flip Image
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                
                # 2. Flip Grid (The Math part)
                # Reshape to 10x10 -> Flip Upside Down -> Flatten back to 100
                grid_2d = risk_vector.reshape(10, 10)
                grid_flipped = np.flipud(grid_2d)
                risk_vector = grid_flipped.flatten()

            # 50% chance to flip HORIZONTALLY
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                grid_2d = risk_vector.reshape(10, 10)
                grid_flipped = np.fliplr(grid_2d)
                risk_vector = grid_flipped.flatten()

        # Apply standard transforms (Resize, Tensor conversion)
        if self.transform:
            image = self.transform(image)
            
        return image, torch.from_numpy(risk_vector.copy()) # .copy() avoids negative stride errors

# --- 2. MODEL (ResNet18) ---
class CrowdRiskNet(nn.Module):
    def __init__(self):
        super(CrowdRiskNet, self).__init__()
        # Load Pretrained Weights
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Higher dropout for robustness
            nn.Linear(512, 100)
        )

    def forward(self, x): return self.backbone(x)

# --- 3. LOSS FUNCTION ---
class RobustLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Moderate Penalties (Balanced)
        self.weights = torch.tensor([1.0, 5.0, 10.0, 15.0, 30.0]).to(device)

    def forward(self, pred, target):
        squared_diff = (pred - target) ** 2
        target_indices = torch.clamp(torch.round(target), 0, 4).long()
        weight_map = self.weights[target_indices]
        return torch.mean(squared_diff * weight_map)

# --- 4. TRAINING LOOP ---
def train_system():
    # Standard transforms (No flipping here, we do it manually above)
    base_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3), # Stronger lighting variance
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Initializing Robust Training...")
    # Enable 'augment=True' for training
    train_dataset = CrowdRiskDataset(DATASET_ROOT, 'train', base_transforms, augment=True)
    val_dataset = CrowdRiskDataset(DATASET_ROOT, 'val', val_transforms, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = CrowdRiskNet().to(device)
    criterion = RobustLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {len(train_dataset)} images with Random Flips (V/H).")

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
    print("\nTraining Complete! 'crowd_risk_model.pth' updated.")

if __name__ == "__main__":
    train_system()