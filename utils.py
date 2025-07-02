# utils.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import random
from config import FIGURE_DIR, BASIN_LIST_FILE, TEST_FOLD_INDEX
#later add nse making function also in the utils

basin_df = pd.read_csv(BASIN_LIST_FILE, dtype=str)
basin_df['fold'] = basin_df['fold'].astype(int)
pub_basin_ids = basin_df[basin_df['fold'] == TEST_FOLD_INDEX]['basin'].tolist()


def load_data(folder, input_dim, seq_len, test_fold_idx=TEST_FOLD_INDEX):
    basin_df = pd.read_csv(BASIN_LIST_FILE, dtype=str)
    basin_df['fold'] = basin_df['fold'].astype(int)

    test_basins = basin_df[basin_df['fold'] == test_fold_idx]['basin'].tolist()
    train_basins = basin_df[basin_df['fold'] != test_fold_idx]['basin'].tolist()
    all_basins = train_basins + test_basins

    scaler = StandardScaler()
    all_data, train_targets, test_targets = {}, {}, {}

    # Fit scaler only on training basins
    for basin in train_basins:
        df = pd.read_csv(f"{folder}/lstm_input_{basin}.csv")
        X = df.drop(columns=["Year", "Month", "Day", "q"]).values
        scaler.partial_fit(X)

    # Prepare data for all basins
    for basin in all_basins:
        df = pd.read_csv(f"{folder}/lstm_input_{basin}.csv")
        X = df.drop(columns=["Year", "Month", "Day", "q"]).values
        Y = df["q"].values

        xseqs, yseqs = [], []
        for i in range(len(X) - seq_len):
            x = X[i:i + seq_len]
            y = Y[i + seq_len - 1]
            x_norm = scaler.transform(x)
            xseqs.append(x_norm)
            yseqs.append(y)

        x_tensor = torch.tensor(np.array(xseqs), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(yseqs), dtype=torch.float32)

        all_data[basin] = x_tensor
        train_targets[basin] = y_tensor if basin in train_basins else torch.full_like(y_tensor, float('nan'))
        test_targets[basin] = torch.full_like(y_tensor, float('nan')) if basin in train_basins else y_tensor

    return all_data, train_targets, test_targets #same input for both train and test, but targets differ



def shuffle_train_data(train_data, train_targets):
    """
    Randomly shuffles samples within each basin and also shuffles basin order.
    """
    shuffled_data, shuffled_targets = {}, {}
    basin_ids = list(train_data.keys())
    random.shuffle(basin_ids)
    for b in basin_ids:
        x, y = train_data[b], train_targets[b]
        perm = torch.randperm(x.size(0))
        shuffled_data[b] = x[perm]
        shuffled_targets[b] = y[perm]
    return shuffled_data, shuffled_targets

def save_predictions_per_basin_with_dates(input_folder, output_folder, prefix, basin_ids, pred_tensor_list, seq_len):
    for i, basin in enumerate(basin_ids):
        df = pd.read_csv(f"{input_folder}/lstm_input_{basin}.csv")
        dates = pd.to_datetime(df[['Year', 'Month', 'Day']])
        valid_idx = np.arange(seq_len - 1, len(df))

        preds = pred_tensor_list[i]
        #round predictions to 3 decimal places
        preds = np.round(preds, 3) 
        trues = df['q'].values[seq_len - 1:]  # Use original target values starting from seq_len - 1

        n = min(len(valid_idx), len(preds), len(trues))
        out_df = pd.DataFrame({
            "date": dates.iloc[valid_idx[:n]].values,
            "observed": trues[:n],
            "predicted": preds[:n]
        })
        out_df.to_csv(f"{output_folder}/{prefix}{TEST_FOLD_INDEX}_{basin}.csv", index=False)
        

def plot_loss(train_losses, test_losses, filename=f"{TEST_FOLD_INDEX}loss_curve.png"):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()

def plot_attention(attn, basin_ids, filename=f"{TEST_FOLD_INDEX}attention_heatmap.png", highlight_ids=pub_basin_ids):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(attn.detach().cpu().numpy(), cmap="viridis",
                     xticklabels=basin_ids, yticklabels=basin_ids)
    plt.title("Cross-Basin Attention")
    plt.xlabel("Source Basin")
    plt.ylabel("Target Basin")
    plt.xticks(rotation=90)
    if highlight_ids:
        xtick_labels = ax.get_xticklabels()
        ytick_labels = ax.get_yticklabels()
        for i, b in enumerate(basin_ids):
            if b in highlight_ids:
                xtick_labels[i].set_color("red")
                ytick_labels[i].set_color("red")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()