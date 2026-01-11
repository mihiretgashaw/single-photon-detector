import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
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
        # Sigmoid keeps weights between 0 and 1 to act as a feature gate
        g_emb = torch.sigmoid(self.gain_fc(g)).view(-1, 256, 1, 1)
        b = b * g_emb
        
        # Upward path with skip connections (torch.cat)
        d2 = self.dec2(torch.cat([self.up2(b), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        
        # Residual connection: input x + learned noise correction
        return torch.clamp(x + self.final(d1), 0, 1)

# Inference function
def run_inference(image_path, model_path, ppp_level=5.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = QIS_UNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Weights loaded successfully.")
    else:
        print("Warning: Model weights not found. Using untrained model.")
    model.eval()
    
    # Load and Preprocess Image
    img = Image.open(image_path).convert('L')
    transform = transforms.ToTensor()
    clean_tensor = transform(img).unsqueeze(0).to(device)
    
    # Generate Noisy Input
    gain_calc = GainCalculator()
    # Fixed: Passing clean_tensor and ppp_level separately to match new forward model
    noisy_tensor, norm_gain = torch_forward_model(clean_tensor, ppp_level, gain_calc)
    
    # Denoise
    with torch.no_grad():
        # norm_gain needs an extra dimension for batch (unsqueeze)
        denoised_tensor = model(noisy_tensor.to(device), norm_gain.unsqueeze(0).to(device))
    
    # Visualization
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(clean_tensor[0, 0].cpu(), cmap='gray')
    plt.title("Original (Ground Truth)")
    
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_tensor[0, 0].cpu(), cmap='gray')
    plt.title(f"Noisy PPP={ppp_level}")
    
    plt.subplot(1, 3, 3)
    plt.imshow(denoised_tensor[0, 0].cpu(), cmap='gray')
    plt.title("Denoised (U-Net Output)")
    plt.show()

if __name__ == "__main__":
    # Internal test to ensure shapes are correct
    test_img = torch.randn(1, 1, 128, 128)
    test_gain = torch.tensor([[0.5]])
    net = QIS_UNet()
    out = net(test_img, test_gain)
    print(f"Test Successful. Output shape: {out.shape}")