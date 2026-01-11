import sys
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import csv

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

# Custom Modules (Ensure these files exist in your parent directory)
from model import QIS_UNet
from dataset import GainCalculator, torch_forward_model
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

# ----------------------------
# Utility Functions
# ----------------------------
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return torch.tensor(float('inf'))
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def poisson_nll_eval(pred, target, ppp):
    pred_ph = torch.clamp(pred * ppp, 1e-8)
    tgt_ph = target * ppp
    return torch.mean(pred_ph - tgt_ph * torch.log(pred_ph))

def save_sharp_image(data, path, title, cmap='gray', vmin=None, vmax=None, label=None):
    """Saves an independent, high-resolution image with its own scale."""
    plt.figure(figsize=(8, 8))
    # interpolation='none' or 'nearest' ensures 'jots' don't get blurry
    im = plt.imshow(data, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
    
    # Add an independent colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    if label:
        cbar.set_label(label)
        
    plt.title(title)
    plt.axis('off')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

# ----------------------------
# Evaluation Function
# ----------------------------
def run_full_evaluation(img_path, weights_path="../checkpoints/qis_master.pth",
                        ppp_levels=[0.5, 1.0, 3.0, 7.0], use_poisson=False):
    
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = QIS_UNet().to(device)
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load Image
    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} not found.")
        return

    img = Image.open(img_path).convert('L')
    clean = transforms.ToTensor()(img).unsqueeze(0).to(device)
    clean_np = clean[0,0].cpu().numpy()

    gain_calc = GainCalculator()
    results_for_csv = []

    print(f"\n{'PPP':<8} | {'Noisy PSNR':<12} | {'Denoised PSNR':<14} | {'SSIM':<8}")
    print("-" * 65)

    # ----------------------------
    # Evaluation Loop
    # ----------------------------
    for ppp in ppp_levels:
        # 1. Forward Model (Generation)
        noisy, norm_gain = torch_forward_model(clean, ppp, gain_calc)
        
        # 2. Inference
        with torch.no_grad():
            denoised = model(noisy.to(device), norm_gain.unsqueeze(0).to(device))
        denoised = torch.clamp(denoised, 0, 1)

        # 3. Metrics Calculation
        psnr_noisy = calculate_psnr(clean, noisy)
        psnr_denoised = calculate_psnr(clean, denoised)
        ssim_val = ssim(denoised, clean, data_range=1.0)
        
        # 4. Prepare NumPy arrays for plotting
        noisy_np = noisy[0,0].cpu().numpy()
        restored_np = denoised[0,0].cpu().numpy()
        abs_error_map = np.abs(clean_np - restored_np)

        # 5. Save Independent Sharp Images
        # Save Noisy
        save_sharp_image(noisy_np, 
                         os.path.join(results_dir, f"noisy_ppp_{ppp}.png"), 
                         f"Noisy Input (PPP {ppp})", cmap='gray')
        
        # Save Restored
        save_sharp_image(restored_np, 
                         os.path.join(results_dir, f"restored_ppp_{ppp}.png"), 
                         f"Restored Output (PPP {ppp})", cmap='gray')
        
        # Save Error Map (Independent Scale)
        save_sharp_image(abs_error_map, 
                         os.path.join(results_dir, f"error_map_ppp_{ppp}.png"), 
                         f"Abs Error Map (PPP {ppp})", cmap='hot', label='Error Intensity')

        # 6. Logging & CSV
        print(f"{ppp:<8} | {psnr_noisy:>10.2f} dB | {psnr_denoised:>12.2f} dB | {ssim_val:>8.4f}")
        
        csv_entry = {
            "PPP": ppp,
            "Noisy_PSNR": round(float(psnr_noisy), 2),
            "Denoised_PSNR": round(float(psnr_denoised), 2),
            "SSIM": round(float(ssim_val), 4)
        }
        if use_poisson:
            poisson_val = poisson_nll_eval(denoised, clean, ppp)
            csv_entry["Poisson_NLL"] = "{:.4e}".format(float(poisson_val))
        results_for_csv.append(csv_entry)

    # ----------------------------
    # Save Final Metrics Data
    # ----------------------------
    # CSV
    csv_filename = os.path.join(results_dir, 'evaluation_results.csv')
    fieldnames = list(results_for_csv[0].keys())
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_for_csv)

    # Performance Plot
    plt.figure(figsize=(10,6))
    ppp_vals = [res["PPP"] for res in results_for_csv]
    plt.plot(ppp_vals, [r["Noisy_PSNR"] for r in results_for_csv], 'ro--', label='Noisy Input')
    plt.plot(ppp_vals, [r["Denoised_PSNR"] for r in results_for_csv], 'bs-', label='U-Net Restored')
    plt.xlabel('Photons Per Pixel (PPP)')
    plt.ylabel('PSNR (dB)')
    plt.title("Restoration Performance: PSNR vs PPP")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "performance_graph.png"), dpi=300)
    plt.close()

    print(f"\nSuccess! All independent sharp images and CSV saved in: {os.path.abspath(results_dir)}")

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # Ensure this path points to your actual test image
    TEST_IMAGE = "../images/my_test_image.jpg" 
    run_full_evaluation(TEST_IMAGE, weights_path="../checkpoints/qis_master.pth", use_poisson=True)