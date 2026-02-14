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
EPOCHS = 20  
LEARNING_RATE = 0.0001
SAVE_NAME = "crowd_risk_model.pth" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DATASET LOADER ---
class CrowdRiskDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, augment=False):
        self.split_dir = os.path.join(root_dir, split)
        self.lbl_dir = os.path.join(self.split_dir, 'labels')
        self.transform = transform
        self.augment = augment
        self.file_ids = [f.replace('.json', '') for f in os.listdir(self.lbl_dir) if f.endswith('.json')]

    def __len__(self): return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        with open(os.path.join(self.lbl_dir, file_id + '.json'), 'r') as f:
            data = json.load(f)
            risk_vector = np.array(data['grid_vector'], dtype=np.float32)

        img_path = os.path.join(self.split_dir, 'visualizations', file_id + '.png')
        image = Image.open(img_path).convert('RGB')
        
        # SMART AUGMENTATION (Flips image AND grid to fix bias)
        if self.augment:
            if random.random() > 0.5: # Vertical Flip
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                risk_vector = np.flipud(risk_vector.reshape(10, 10)).flatten()
            if random.random() > 0.5: # Horizontal Flip
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                risk_vector = np.fliplr(risk_vector.reshape(10, 10)).flatten()

        if self.transform: image = self.transform(image)
        return image, torch.from_numpy(risk_vector.copy())

# --- 2. MODEL ---
class CrowdRiskNet(nn.Module):
    def __init__(self):
        super(CrowdRiskNet, self).__init__()
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 100)
        )
    def forward(self, x): return self.backbone(x)

# --- 3. TRAINING LOOP ---
def train_new_model():
    print(f"Creating new model on: {device}")
    
    transforms_train = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Data
    train_dataset = CrowdRiskDataset(DATASET_ROOT, 'train', transforms_train, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    model = CrowdRiskNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Simple Mean Squared Error for robust learning

    print(f"Starting training for {EPOCHS} epochs...")

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
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f}")

    # SAVE THE FILE
    torch.save(model.state_dict(), SAVE_NAME)
    print(f"\nSUCCESS: New model saved as '{SAVE_NAME}'")

if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT):
        train_new_model()
    else:
        print(f"Error: {DATASET_ROOT} not found. Run prepare_dataset.py first.")