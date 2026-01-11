import sys
import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import csv

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

# Custom Modules
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

# ----------------------------
# Evaluation Function
# ----------------------------
def run_full_evaluation(img_path, weights_path="../checkpoints/qis_master.pth",
                        ppp_levels=[0.5,1.0,3.0,7.0], use_poisson=False):
    # ----------------------------
    # Folders
    # ----------------------------
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # ----------------------------
    # Device & Model
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = QIS_UNet().to(device)
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # ----------------------------
    # Load Image
    # ----------------------------
    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} not found.")
        return

    img = Image.open(img_path).convert('L')
    clean = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # ----------------------------
    # Gain Calculator
    # ----------------------------
    gain_calc = GainCalculator()
    results_for_csv = []

    # ----------------------------
    # Visualization Grid using GridSpec
    # ----------------------------
    n_rows = len(ppp_levels)
    n_cols = 4
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.05, wspace=0.05)

    print(f"\n{'PPP':<8} | {'Noisy PSNR':<12} | {'Denoised PSNR':<14} | {'SSIM':<8}")
    print("-" * 65)

    # ----------------------------
    # Evaluation Loop
    # ----------------------------
    for i, ppp in enumerate(ppp_levels):
        # Generate noisy image and gain
        noisy, norm_gain = torch_forward_model(clean, ppp, gain_calc)
        with torch.no_grad():
            denoised = model(noisy.to(device), norm_gain.unsqueeze(0).to(device))
        denoised = torch.clamp(denoised, 0, 1)

        # Metrics
        psnr_noisy = calculate_psnr(clean, noisy)
        psnr_denoised = calculate_psnr(clean, denoised)
        ssim_val = ssim(denoised, clean, data_range=1.0)
        poisson_val = poisson_nll_eval(denoised, clean, ppp) if use_poisson else None

        # Absolute Error Map
        abs_error_map = torch.abs(clean - denoised) * 5.0
        abs_error_map = torch.clamp(abs_error_map, 0, 0.1)

        # Log-scaled Error Map
        log_error_map = torch.log1p(torch.abs(clean - denoised) * 10)
        log_error_map = log_error_map / log_error_map.max()

        # Print metrics
        print(f"{ppp:<8} | {psnr_noisy:>10.2f} dB | {psnr_denoised:>12.2f} dB | {ssim_val:>8.4f}")

        # CSV entry
        csv_entry = {
            "PPP": ppp,
            "Noisy_PSNR": round(float(psnr_noisy), 2),
            "Denoised_PSNR": round(float(psnr_denoised), 2),
            "SSIM": round(float(ssim_val), 4)
        }
        if use_poisson:
            # csv_entry["Poisson_NLL"] = round(float(poisson_val), 6)
            csv_entry["Poisson_NLL"] = "{:.4e}".format(float(poisson_val))
        results_for_csv.append(csv_entry)

        # ----------------------------
        # Plotting
        # ----------------------------
        ax_orig = fig.add_subplot(gs[i,0])
        ax_noisy = fig.add_subplot(gs[i,1])
        ax_denoised = fig.add_subplot(gs[i,2])
        ax_error = fig.add_subplot(gs[i,3])

        ax_orig.imshow(clean[0,0].cpu(), cmap='gray'); ax_orig.axis('off'); ax_orig.set_title("Original")
        ax_noisy.imshow(noisy[0,0].cpu(), cmap='gray'); ax_noisy.axis('off'); ax_noisy.set_title(f"Noisy (PPP {ppp})")
        ax_denoised.imshow(denoised[0,0].cpu(), cmap='gray'); ax_denoised.axis('off'); ax_denoised.set_title("Restored")
        im_err = ax_error.imshow(abs_error_map[0,0].cpu(), cmap='hot')
        ax_error.axis('off'); ax_error.set_title("Abs Error Map")
        fig.colorbar(im_err, ax=ax_error, fraction=0.03, pad=0.01)

        # Save log-scaled error map separately
        log_map_path = os.path.join(results_dir, f"log_error_map_ppp_{ppp}.png")
        plt.imsave(log_map_path, log_error_map[0,0].cpu().numpy(), cmap='inferno')

    # ----------------------------
    # Save CSV
    # ----------------------------
    csv_filename = os.path.join(results_dir, 'evaluation_results.csv')
    fieldnames = ["PPP", "Noisy_PSNR", "Denoised_PSNR", "SSIM"]
    if use_poisson:
        fieldnames.append("Poisson_NLL")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_for_csv)

    # ----------------------------
    # Performance Graph
    # ----------------------------
    plt.figure(figsize=(10,6))
    ppp_vals = [res["PPP"] for res in results_for_csv]
    plt.plot(ppp_vals, [r["Noisy_PSNR"] for r in results_for_csv], 'ro--', label='Noisy Input')
    plt.plot(ppp_vals, [r["Denoised_PSNR"] for r in results_for_csv], 'bs-', label='U-Net Restored')
    plt.xlabel('Photons Per Pixel (PPP)')
    plt.ylabel('PSNR (dB)')
    plt.title("PSNR vs PPP")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "performance_graph.png"))

    # Save main grid
    fig.savefig(os.path.join(results_dir, "full_evaluation_grid.png"), bbox_inches='tight', dpi=300)
    plt.tight_layout()
    print(f"\nSuccess! CSV, Graph, and Error Maps saved in '{results_dir}' folder.")
    plt.show()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    TEST_IMAGE = "../images/my_test_image.jpg"  # Replace with your test image
    run_full_evaluation(TEST_IMAGE, weights_path="../checkpoints/qis_master.pth", use_poisson=True)
