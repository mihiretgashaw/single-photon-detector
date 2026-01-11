import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# --- PATH FIX ---
# This allows model.py to find dataset.py even if run from inside the /scripts folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from dataset import GainCalculator, torch_forward_model 

class QIS_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def cb(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(o, o, 3, 1, 1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2, True)
            )
        # Encoder
        self.enc1 = cb(1, 64)
        self.enc2 = cb(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = cb(128, 256)
        
        # Gain Conditioning Layer
        self.gain_fc = nn.Linear(1, 256)
        
        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = cb(256 + 128, 128)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = cb(128 + 64, 64)
        
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x, g):
        # Downward path
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        
        # Bottleneck
        b = self.bottleneck(self.pool(s2))
        
        # Apply Gain Conditioning
        g_emb = torch.sigmoid(self.gain_fc(g)).view(-1, 256, 1, 1)
        b = b * g_emb
        
        # Upward path with skip connections
        d2 = self.dec2(torch.cat([self.up2(b), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        
        # Residual connection
        return torch.clamp(x + self.final(d1), 0, 1)

# Inference function for quick single-image testing
def run_single_inference(image_path, model_path="../checkpoints/qis_master.pth", ppp_level=5.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = QIS_UNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Weights loaded from {model_path}")
    else:
        print("Warning: Weights not found. Using untrained model.")
    model.eval()
    
    img = Image.open(image_path).convert('L')
    clean_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    gain_calc = GainCalculator()
    noisy_tensor, norm_gain = torch_forward_model(clean_tensor, ppp_level, gain_calc)
    
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor.to(device), norm_gain.unsqueeze(0).to(device))
    
    plt.figure(figsize=(12, 4))
    titles = ["Ground Truth", f"Noisy (PPP {ppp_level})", "Denoised Output"]
    imgs = [clean_tensor, noisy_tensor, denoised_tensor]
    
    for i, (im, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, 3, i+1)
        plt.imshow(im[0, 0].cpu(), cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Test shape consistency
    test_img = torch.randn(1, 1, 128, 128)
    test_gain = torch.tensor([[0.5]])
    net = QIS_UNet()
    out = net(test_img, test_gain)
    print(f"Structure Validated. Output shape: {out.shape}")