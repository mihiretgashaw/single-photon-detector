import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms

class GainCalculator:
    def compute(self, ppp):
        # Normalizes Photons Per Pixel to [0, 1] for the model
        # Using 200.0 as the 'max' reference for normalization
        return torch.tensor([ppp / 200.0])

def torch_forward_model(img_tensor, ppp, gain_calc, read_noise_std=0.0):
    # 1. Poisson Sampling (Shot Noise)
    # img_tensor is ground truth [0, 1], ppp is mean photon flux
    noisy = torch.poisson(img_tensor.clamp(0, 1) * ppp)
    
    # 2. Add Read Noise (Optional Gaussian noise)
    if read_noise_std > 0:
        noisy += torch.randn_like(noisy) * read_noise_std
        
    # 3. Normalize back to [0, 1] for neural network input
    noisy = noisy / ppp
    
    norm_gain = gain_calc.compute(ppp)
    return noisy, norm_gain

class QISDataset(Dataset):
    def __init__(self, root_dir, ppp_value=None, is_train=True):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Training uses random crops; Validation uses center crops or resize
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
        
        # If ppp_value is None, we randomize (Training)
        # If ppp_value is set, we use that fixed value (Validation)
        ppp = self.ppp_value if self.ppp_value is not None else np.random.uniform(0.5, 200.0)
        
        noisy, norm_gain = torch_forward_model(clean, ppp, self.gain_calc)
        
        return noisy, clean, norm_gain.view(1)