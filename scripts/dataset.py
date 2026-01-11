import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms
import sys
from scipy.interpolate import Rbf

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

class GainCalculator:
    def __init__(self):
        # This is your original scientific sensor data
        self.data = np.array([
            [0.5, 90], [1.5, 60], [2.5, 50], [3.25, 30], [6.5, 15], [9.75, 7.5],
            [13, 4.5], [20, 3.2], [26, 2.8], [36, 2.4], [45, 2.2], [54, 1.8],
            [67, 1.5], [80, 1.3], [90, 1.1], [110, 1.05], [130, 0.9], [145, 0.65],
            [155, 0.56], [160, 0.51], [200, 0.4881704]
        ])
        # Restoration of your RBF interpolation
        self.rbf = Rbf(self.data[:, 0], self.data[:, 1], function='linear')

    def compute(self, avg_PPP):
        # Returns the interpolated gain value based on light level
        gain = self.rbf(avg_PPP)
        # Normalize the gain (usually 0-1) for the Neural Network input
        return torch.tensor([gain / 90.0], dtype=torch.float32)

def torch_forward_model(img_tensor, ppp, gain_calc):
    # Poisson Sampling (Physics-based shot noise)
    noisy = torch.poisson(img_tensor.clamp(0, 1) * ppp)
    
    # Normalize back to 0-1 range
    noisy = noisy / ppp
    
    # Calculate the non-linear gain using your RBF logic
    norm_gain = gain_calc.compute(ppp)
    return noisy, norm_gain

class QISDataset(Dataset):
    def __init__(self, root_dir, ppp_value=None, is_train=True):
        if not os.path.isabs(root_dir) and not os.path.exists(root_dir):
             root_dir = os.path.join(parent_dir, root_dir)
             
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(128),
                transforms.ToTensor()
            ])
            
        self.ppp_value = ppp_value
        self.gain_calc = GainCalculator()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        clean = self.transform(Image.open(self.files[idx]).convert('L'))
        
        # FOCUSING ON DARK ENVIRONMENT: 1.5 PPP
        ppp = self.ppp_value if self.ppp_value is not None else 1.5
        
        noisy, norm_gain = torch_forward_model(clean, ppp, self.gain_calc)
        
        return noisy, clean, norm_gain.view(1)

if __name__ == "__main__":
    print("Testing Dataset with RBF Gain Calculator...")
    gc = GainCalculator()
    # Test your RBF at 1.5 PPP
    print(f"Gain at 1.5 PPP: {gc.compute(1.5).item() * 90:.2f}") # Should be 60.0