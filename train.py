# train.py

import os
import torch
import numpy as np
import pandas as pd
import random
from config import hparams, DATA_FOLDER, OUTPUT_DIR, FIGURE_DIR, BASIN_LIST_FILE, DEVICE, MODEL_CKPT_FILE, ATTENTION_MATRIX_FILE
from model import BasinLevelCrossBasinAttention
from utils import (
    load_data, save_predictions_per_basin_with_dates,
    plot_loss, plot_attention, shuffle_train_data
)

def train_and_evaluate(model, all_data, train_targets, device, folder, seq_len):
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    loss_fn = torch.nn.MSELoss()
    train_losses = []

    # Deep copy for use during prediction after training
    all_data_orig = {b: v.clone() for b, v in all_data.items()}

    for epoch in range(hparams['num_epochs']):
        model.train()
        optimizer.zero_grad()

        train_data, train_targets = shuffle_train_data(all_data, train_targets)
        X_b = [train_data[b].to(device) for b in train_data]
        Y_b = [train_targets[b].to(device) for b in train_targets]

        pred, attn = model(X_b)
        mask = ~torch.isnan(torch.cat(Y_b))
        loss = loss_fn(torch.cat(pred)[mask], torch.cat(Y_b)[mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}: Train Loss = {loss.item():.4f}")

    # Save trained model and attention matrix
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, MODEL_CKPT_FILE))
    torch.save(attn, os.path.join(OUTPUT_DIR, ATTENTION_MATRIX_FILE))

    # Inference on all basins
    model.eval()
    with torch.no_grad():
        X_all = [all_data_orig[b].to(device) for b in all_data_orig]
        ids_all = list(all_data_orig.keys())
        pred_all, attn = model(X_all)
        pred_seq = [p.cpu().numpy().tolist() for p in pred_all]
        save_predictions_per_basin_with_dates(folder, f"fold", ids_all, pred_seq, seq_len)

    plot_loss(train_losses)
    plot_attention(attn, ids_all)

    print(f"Total Basins: {len(ids_all)}")
    print(f"Attention Matrix Size: {attn.size()}")
    return model


# Main entry point for standalone runs
if __name__ == "__main__":
    basin_list = pd.read_csv(BASIN_LIST_FILE, dtype=str)['basin'].tolist()

    model = BasinLevelCrossBasinAttention(
        input_dim=hparams['input_dim'],
        hidden_dim=hparams['hidden_dim'],
        num_layers=hparams['num_layers'],
        dropout=hparams['dropout'],
        context_dropout=hparams['context_dropout'],
        num_heads=hparams['num_heads']
    ).to(DEVICE)

    all_data, train_targets, _ = load_data(
        DATA_FOLDER, basin_list, hparams['input_dim'], hparams['seq_length']
    )

    train_and_evaluate(
        model, all_data, train_targets,
        DEVICE, DATA_FOLDER,
        seq_len=hparams['seq_length']
    )
