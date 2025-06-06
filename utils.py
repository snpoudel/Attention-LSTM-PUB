# utils.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import random
from config import FIGURE_DIR, OUTPUT_DIR, BASIN_LIST_FILE

#later add nse making function also in the utils

pub_basin_ids = pd.read_csv(BASIN_LIST_FILE, dtype=str).query("gauge == 'False'")['basin'].tolist()
#NEED TO MAKE PUB IDS AS THOSE WHOSE FOLD INDEX IS HELD OUT FOR VALIDATION

def load_data(folder, basin_list, input_dim, seq_len):
    scaler = StandardScaler()
    train_data, test_data, train_targets, test_targets = {}, {}, {}, {}

    # Fit scaler
    for basin in basin_list:
        df = pd.read_csv(f"{folder}/lstm_input_{basin}.csv")
        df_train = df[df["Year"] <= 2010]
        X = df_train.drop(columns=["Year", "Month", "Day", "q"]).values
        scaler.partial_fit(X)

    for basin in basin_list:
        df = pd.read_csv(f"{folder}/lstm_input_{basin}.csv")
        years = df["Year"].values
        X = df.drop(columns=["Year", "Month", "Day", "q"]).values
        Y = df["q"].values

        xseqs_train, yseqs_train, xseqs_test, yseqs_test = [], [], [], []

        for i in range(len(X) - seq_len):
            x = X[i:i+seq_len]
            y = Y[i+seq_len-1]
            yr = years[i+seq_len-1]
            x_norm = scaler.transform(x)
            if yr <= 2010:
                xseqs_train.append(x_norm)
                yseqs_train.append(y)
            else:
                xseqs_test.append(x_norm)
                yseqs_test.append(y)

        if xseqs_train:
            train_data[basin] = torch.tensor(xseqs_train, dtype=torch.float32)
            train_targets[basin] = torch.tensor(yseqs_train, dtype=torch.float32)
        if xseqs_test:
            test_data[basin] = torch.tensor(xseqs_test, dtype=torch.float32)
            test_targets[basin] = torch.tensor(yseqs_test, dtype=torch.float32)

    return train_data, test_data, train_targets, test_targets

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

def save_predictions_per_basin_with_dates(folder, prefix, basin_ids, pred_tensor_list, target_tensor_list, seq_len):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, basin in enumerate(basin_ids):
        df = pd.read_csv(f"{folder}/lstm_input_{basin}.csv")
        dates = pd.to_datetime(df[['Year', 'Month', 'Day']])
        valid_idx = np.arange(seq_len - 1, len(df))

        preds = pred_tensor_list[i]
        trues = target_tensor_list[i]

        n = min(len(valid_idx), len(preds), len(trues))
        out_df = pd.DataFrame({
            "date": dates.iloc[valid_idx[:n]].values,
            "observed": trues[:n],
            "predicted": preds[:n]
        })
        out_df.to_csv(f"{OUTPUT_DIR}/{prefix}_{basin}.csv", index=False)
        

def plot_loss(train_losses, test_losses, filename="loss_curve.png"):
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

def plot_attention(attn, basin_ids, filename="attention_heatmap.png", highlight_ids=pub_basin_ids):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(attn.cpu().numpy(), cmap="viridis",
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