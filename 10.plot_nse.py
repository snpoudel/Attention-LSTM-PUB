import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#version of the model to generate the NSEs
ver = 'logtemp'
# --------------------------
# Load Basin List
# --------------------------
basin_list = pd.read_csv("data/basin_list_with_folds.csv", dtype=str)
ungauged_ids = basin_list[basin_list['fold'] == '1']['basin'].tolist()
gauged_ids = basin_list[basin_list['fold'] != '1']['basin'].tolist()

# --------------------------
# NSE Function
# --------------------------
def nse(pred, obs):
    mean_obs = np.mean(obs)
    return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - mean_obs) ** 2)

# --------------------------
# Function to Compute NSE Records
# --------------------------
def compute_nse_records(basins, model_output_dir, model_name, file_index):
    records = []
    for basin in basins:
        try:
            df = pd.read_csv(f"{model_output_dir}/{file_index}{basin}.csv")
            pred = df['predicted'].values
            obs = df['observed'].values
            nse_value = nse(pred, obs)
            # gauge_status = 'Gauged' if basin in gauged_ids else 'Ungauged'
            records.append({'basin': basin, 'nse': nse_value, 'model': model_name})
        except Exception as e:
            print(f"Error with basin {basin}: {e}")
    return records

# --------------------------
# Compute NSEs for Both Models
# --------------------------
records_cba = compute_nse_records(ungauged_ids, "output", "CBA-LSTM", "fold1_")
records_vanilla = compute_nse_records(ungauged_ids, "output/vanilla_lstm/", "Vanilla LSTM", "fold1_")

# Combine DataFrames
df_nse_all = pd.DataFrame(records_cba + records_vanilla)

# --------------------------
# Plot CDF for Both Models
# --------------------------
# Plot CDF for Both Models with color by model, marker by gauge type
plt.figure(figsize=(6, 4))
# Assign a color to each model
model_colors = {
    'CBA-LSTM': 'tab:blue',
    'Vanilla LSTM': 'tab:orange',
}
medians = {}
for model_name in df_nse_all['model'].unique():
    subset = df_nse_all[(df_nse_all['model'] == model_name)]
    sorted_nse = np.sort(subset['nse'])
    cdf = np.linspace(0, 1, len(sorted_nse))
    color = model_colors[model_name]
    # Line
    plt.plot(sorted_nse, cdf, color=color, label=model_name)
    # Points with custom marker
    plt.scatter(sorted_nse, cdf, s=10, color=color, alpha=0.7)
    # Median value
    median_nse = np.median(subset['nse'])
    medians[model_name] = median_nse

# Show median NSEs as text in the figure (top left corner)
median_text = "\n".join([f"Median {model}: {medians[model]:.2f}" for model in medians])
plt.text(0.02, 0.8, median_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.xlabel('NSE')
plt.ylabel('Empirical CDF')
plt.title('PUBs with Attention-LSTM vs Vanilla LSTM')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"figures/{ver}nse_combined.png", dpi=300)
plt.show()
