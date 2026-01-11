import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import QISDataset
from model import QIS_UNet
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QIS_UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=4e-4)

# Ensure this path matches your folder exactly
data_path = "DIV2K_train_HR"
dataset = QISDataset(data_path)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"Starting training on {device}...")

for epoch in range(25):
    model.train()
    total_loss = 0
    for noisy, clean, gain in loader:
        noisy, clean, gain = noisy.to(device), clean.to(device), gain.to(device)
        
        preds = model(noisy, gain)
        loss = F.l1_loss(preds, clean)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"---> Checkpoint saved: {checkpoint_path}")

# # Final save
# torch.save(model.state_dict(), "qis_master.pth")
# print("Training Complete. Final model saved as qis_master.pth")

# Save into the 'checkpoints' folder
torch.save(model.state_dict(), "checkpoints/qis_master.pth")
print("Training Complete. Final model saved in checkpoints/qis_master.pth")