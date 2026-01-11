# Single-Photon Detector (SPD) Image Restoration

This project implements a **Residual U-Net** designed to restore images from a Quanta Image Sensor (QIS) environment. It specializes in denoising images at ultra-low light levels (0.1 to 5.0 Photons Per Pixel).

📁 Project Structure

model.py: The Residual U-Net architecture featuring Gain Conditioning.
dataset.py: Handles DIV2K data loading and QIS noise simulation (Poisson + Read Noise).
train.py: The training pipeline with Mean Squared Error (MSE) loss.
stress_test.py: Evaluates the model across 1.0, 3.0, 5.0, and 7.0 PPP levels.
plot_results.py: Generates scientific PSNR performance graphs from CSV data.
inference.py: Quick script for running a single image through the model.
/checkpoints: Stores the trained weights (qis_master.pth).
/results: Stores all evaluation PNGs, grids, and evaluation_results.csv.
/images: Directory for your test input images.
/div2k_data & /DIV2K_train_HR: Training and validation datasets.

## 🚀 How to Runwo`

1. **Setup Environment**:

   ```bash
   pip install torch torchvision matplotlib pandas

   Run Evaluation: To test the trained model on a specific image and see the results across 4 light levels:

             python stress_test.py

   Generate Performance Graph: After running the stress test, generate the scientific PSNR curve:

            python plot_results.py

   🧠 Technical Highlights

   1. Gain Conditioning: The model receives the Normalized Gain (light level) as a secondary input. This allows the U-Net to adapt its weights dynamically depending on whether the input is slightly noisy (5.0 PPP) or extremely dark (0.1 PPP).

   2. Residual Learning: The model predicts a noise correction map rather than the full image. The final output is: Output = Clamp(Input + Learned Correction, 0, 1)

   📊 Results Summary: The system provides significant PSNR (dB) gains in the critical 0.1–1.0 PPP range. While raw sensor data at these levels is often unusable, the Residual U-Net successfully recovers structural details and reduces "salt-and-pepper" photon noise.
   ```
