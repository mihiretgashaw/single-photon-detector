import sys
import os

# --- PATH FIX FOR SUBFOLDER ---
# Adds the parent directory to the search path so it can find model.py and dataset.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Custom Modules (from parent directory)
from dataset import QISDataset
from model import QIS_UNet

def train_model():
    # --- FOLDER SETUP ---
    # Ensure checkpoints folder exists relative to the main project root
    checkpoint_dir = os.path.join(parent_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created directory: {checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QIS_UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=4e-4)

    # Path to dataset (assumes it is in the root folder)
    data_path = os.path.join(parent_dir, "DIV2K_train_HR")
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset folder not found at {data_path}")
        return

    dataset = QISDataset(data_path)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(f"Starting training on {device}...")
    print(f"Data Path: {data_path}")

    for epoch in range(25):
        model.train()
        total_loss = 0
        for noisy, clean, gain in loader:
            noisy, clean, gain = noisy.to(device), clean.to(device), gain.to(device)
            
            preds = model(noisy, gain)
            loss = F.l1_loss(preds, clean) # L1 Loss for better edge preservation
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:02d}/25 | Avg L1 Loss: {avg_loss:.6f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            name = f"checkpoint_epoch_{epoch+1}.pth"
            path = os.path.join(checkpoint_dir, name)
            torch.save(model.state_dict(), path)
            print(f"--- Checkpoint saved: {path}")

    # Final save
    final_path = os.path.join(checkpoint_dir, "qis_master.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining Complete! Final model saved: {final_path}")

if __name__ == "__main__":
    train_model()