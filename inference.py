import torch
from model import QIS_UNet
from dataset import GainCalculator, torch_forward_model
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

def main():
    # Settings - UPDATED PATH
    img_path = "images/my_test_image.jpg"
    weights_path = "checkpoints/qis_master.pth" # Points to your new folder
    ppp = 0.5 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if weights exist before loading
    if not os.path.exists(weights_path):
        print(f"Error: Could not find {weights_path}. Did you move the file to the checkpoints folder?")
        return

    # Load Model
    model = QIS_UNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load Image
    img = Image.open(img_path).convert('L')
    clean = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # Simulate QIS Noise
    gain_calc = GainCalculator()
    noisy, norm_gain = torch_forward_model(clean, ppp, gain_calc)

    # Run Inference
    with torch.no_grad():
        # norm_gain needs to be (Batch, 1) -> unsqueeze(0)
        denoised = model(noisy, norm_gain.unsqueeze(0).to(device))

    # Show Results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(clean[0,0].cpu(), cmap='gray'); ax[0].set_title("Original")
    ax[1].imshow(noisy[0,0].cpu(), cmap='gray'); ax[1].set_title(f"Noisy (PPP={ppp})")
    ax[2].imshow(denoised[0,0].cpu(), cmap='gray'); ax[2].set_title("Denoised")
    
    # Save a quick preview if you want
    plt.savefig("results/quick_test_preview.png")
    print("Inference successful! Preview saved to results/quick_test_preview.png")
    
    plt.show()

if __name__ == "__main__":
    main()