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
BATCH_SIZE = 8          # Reduce to 4 if you run out of memory
LEARNING_RATE = 0.001
EPOCHS = 20             # How many times to loop through the data
IMAGE_SIZE = 512        # We use high res to see small heads

# --- DEVICE SETUP ---
# Uses GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# --- 1. THE DATASET LOADER ---
class CrowdRiskDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        """
        root_dir: 'final_model_dataset'
        split: 'train' or 'val'
        """
        self.split_dir = os.path.join(root_dir, split)
        self.vis_dir = os.path.join(self.split_dir, 'visualizations') # We use the raw images ideally, but visualizations work for demo
        # NOTE: In a real final app, you'd use raw images. 
        # For now, we verify filenames from the label folder.
        self.lbl_dir = os.path.join(self.split_dir, 'labels')
        self.transform = transform
        
        # Get list of valid pairs
        self.file_ids = []
        for f in os.listdir(self.lbl_dir):
            if f.endswith('.json'):
                self.file_ids.append(f.replace('.json', ''))

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        
        # Load JSON Label (The "Answer")
        lbl_path = os.path.join(self.lbl_dir, file_id + '.json')
        with open(lbl_path, 'r') as f:
            data = json.load(f)
            # Get the flattened 100-vector (0-4 risk levels)
            risk_vector = torch.tensor(data['grid_vector'], dtype=torch.float32)

        # Load Image (The "Question")
        # We need to find the original image path. 
        # Since we moved data, we will try to load the visualization image 
        # (which has the grid drawn on it) OR strictly we should load the original raw image.
        # To keep it simple and robust for this hackathon, let's look for the .png we saved.
        img_path = os.path.join(self.split_dir, 'visualizations', file_id + '.png')
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, risk_vector

# --- 2. THE MODEL (Simple CNN) ---
class CrowdRiskNet(nn.Module):
    def __init__(self):
        super(CrowdRiskNet, self).__init__()
        # We use a lightweight ResNet18 backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Replace the last layer to output 100 numbers (10x10 grid)
        # The default resnet output is 1000 classes. We change it to 100 regression values.
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 100) # Output 100 risk scores
        )

    def forward(self, x):
        return self.backbone(x)

# --- 3. TRAINING LOOP ---
def train_system():
    # Transforms (Resize and Normalize)
    data_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Data
    print("Loading datasets...")
    train_dataset = CrowdRiskDataset(DATASET_ROOT, 'train', data_transforms)
    val_dataset = CrowdRiskDataset(DATASET_ROOT, 'val', data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

    # Initialize Model
    model = CrowdRiskNet().to(device)
    
    # Loss Function: MSE (Mean Squared Error) because we are predicting a score (0.0 to 4.0)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- STARTING TRAINING ---")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    # Save the Brain
    torch.save(model.state_dict(), "crowd_risk_model.pth")
    print("\nTraining Complete! Model saved as 'crowd_risk_model.pth'")

if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT):
        train_system()
    else:
        print("Error: Dataset not found. Run prepare_dataset.py first.")