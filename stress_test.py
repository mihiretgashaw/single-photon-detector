import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import csv

# Import your custom modules
from model import QIS_UNet
from dataset import GainCalculator, torch_forward_model

def calculate_psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio between two tensors."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def run_full_evaluation(img_path, weights_path="checkpoints/qis_master.pth"):
    # --- FOLDER SETUP ---
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")

    # 1. Setup Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = QIS_UNet().to(device)
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 2. Load and Prepare Image
    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} not found.")
        return

    img = Image.open(img_path).convert('L')
    clean = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # 3. Define PPP levels to test and Setup CSV
    # ppp_levels = [0.1, 0.5, 1.0, 5.0]
    ppp_levels = [0.5, 1.0, 3.0, 7.0]
    gain_calc = GainCalculator()
    results_for_csv = []

    # Setup Visualization Grid
    fig, axes = plt.subplots(len(ppp_levels), 3, figsize=(15, 20))
    plt.subplots_adjust(hspace=0.4)

    print(f"\n{'PPP':<8} | {'Noisy PSNR':<12} | {'Denoised PSNR':<14} | {'Gain (dB)'}")
    print("-" * 55)

    # 4. Loop through light levels
    for i, ppp in enumerate(ppp_levels):
        # Generate Noisy Input
        noisy, norm_gain = torch_forward_model(clean, ppp, gain_calc)
        
        # Run Model Inference
        with torch.no_grad():
            denoised = model(noisy.to(device), norm_gain.unsqueeze(0).to(device))

        # Calculate Metrics
        psnr_noisy = calculate_psnr(clean, noisy)
        psnr_denoised = calculate_psnr(clean, denoised)
        improvement = psnr_denoised - psnr_noisy

        print(f"{ppp:<8} | {psnr_noisy:>10.2f} dB | {psnr_denoised:>12.2f} dB | +{improvement:.2f} dB")

        # 5. Store data for Table/CSV
        results_for_csv.append({
            "PPP": ppp,
            "Noisy_PSNR": round(float(psnr_noisy), 2),
            "Denoised_PSNR": round(float(psnr_denoised), 2),
            "Improvement_dB": round(float(improvement), 2)
        })

        # 6. Save individual restored images into the RESULTS folder
        restored_pil = transforms.ToPILImage()(denoised.squeeze().cpu())
        img_save_path = os.path.join(results_dir, f"restored_ppp_{ppp}.png")
        restored_pil.save(img_save_path)

        # 7. Plotting onto the Grid
        axes[i, 0].imshow(clean[0,0].cpu(), cmap='gray'); axes[i, 0].set_title("Original")
        axes[i, 1].imshow(noisy[0,0].cpu(), cmap='gray'); axes[i, 1].set_title(f"Noisy (PPP {ppp})")
        axes[i, 2].imshow(denoised[0,0].cpu(), cmap='gray'); axes[i, 2].set_title(f"Denoised ({psnr_denoised:.2f} dB)")
        
        for ax in axes[i]:
            ax.axis('off')

    # 8. Save CSV Table into the RESULTS folder
    csv_filename = os.path.join(results_dir, 'evaluation_results.csv')
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["PPP", "Noisy_PSNR", "Denoised_PSNR", "Improvement_dB"])
        writer.writeheader()
        writer.writerows(results_for_csv)
    print(f"\nNumerical table saved as '{csv_filename}'")

    # 9. Save the entire Grid Image into the RESULTS folder
    grid_filename = os.path.join(results_dir, "full_evaluation_grid.png")
    plt.savefig(grid_filename, bbox_inches='tight', dpi=300)
    print(f"Full grid image saved as '{grid_filename}'")

    plt.show()

if __name__ == "__main__":
    # Ensure this path is correct for your test image
    TEST_IMAGE = "images/my_test_image.jpg"
    run_full_evaluation(TEST_IMAGE)