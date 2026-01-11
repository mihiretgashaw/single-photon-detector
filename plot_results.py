import matplotlib.pyplot as plt
import csv
import os

def plot_performance(csv_path="results/evaluation_results.csv"):
    # Check if the file exists in the results folder
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        print("Make sure you ran stress_test.py and the CSV is in the 'results' folder!")
        return

    ppp, noisy_psnr, denoised_psnr, improvement = [], [], [], []

    # 1. Read the CSV data from the results folder
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ppp.append(float(row['PPP']))
            noisy_psnr.append(float(row['Noisy_PSNR']))
            denoised_psnr.append(float(row['Denoised_PSNR']))
            improvement.append(float(row['Improvement_dB']))

    # 2. Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot Noisy vs Denoised
    plt.plot(ppp, noisy_psnr, 'ro--', label='Raw Noisy Input (Sensor)')
    plt.plot(ppp, denoised_psnr, 'gs-', linewidth=2, markersize=8, label='U-Net Restored')

    # 3. Add Titles and Labels
    plt.title('SPD Model Performance: PSNR Improvement vs. Light Level', fontsize=14)
    plt.xlabel('Photons Per Pixel (PPP) - Darkness Level', fontsize=12)
    plt.ylabel('PSNR (dB) - Higher is Better', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # 4. Annotate the "Gain" (The Improvement)
    for i in range(len(ppp)):
        plt.annotate(f"+{improvement[i]}dB", 
                     (ppp[i], denoised_psnr[i]),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center', 
                     color='green', 
                     fontweight='bold')

    # 5. SAVE the result to the results folder
    save_path = "results/final_performance_graph.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Success! Final graph saved as '{save_path}'")
    
    plt.show()

if __name__ == "__main__":
    plot_performance()