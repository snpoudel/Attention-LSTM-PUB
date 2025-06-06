# train.py

import os
import torch
import numpy as np
import random
from config import hparams, output_dir, figure_dir, model_ckpt_file, attention_matrix_file
from model import BasinLevelCrossBasinAttention
from utils import (
    load_data, save_predictions_per_basin_with_dates,
    plot_loss, plot_attention, shuffle_train_data
)

def train_and_evaluate(model, train_data, train_targets, test_data, test_targets, device, folder, seq_len):
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    loss_fn = torch.nn.MSELoss()
    train_losses, test_losses = [], []
    best_loss = float('inf')

    train_data_orig = {b: v.clone() for b, v in train_data.items()}
    train_targets_orig = {b: v.clone() for b, v in train_targets.items()}

    for epoch in range(hparams['num_epochs']):
        model.train()
        optimizer.zero_grad()

        train_data, train_targets = shuffle_train_data(train_data, train_targets)
        X_b = [train_data[b].to(device) for b in train_data]
        Y_b = [train_targets[b].to(device) for b in train_targets]

        pred, _ = model(X_b)
        mask = ~torch.isnan(torch.cat(Y_b))
        loss = loss_fn(torch.cat(pred)[mask], torch.cat(Y_b)[mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            X_t = [test_data[b].to(device) for b in test_data]
            Y_t = [test_targets[b].to(device) for b in test_targets]
            pred_t, attn = model(X_t)
            mask_t = ~torch.isnan(torch.cat(Y_t))
            val_loss = loss_fn(torch.cat(pred_t)[mask_t], torch.cat(Y_t)[mask_t])
            test_losses.append(val_loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}: Train Loss = {loss.item():.4f}, Test Loss = {val_loss.item():.4f}")

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            torch.save(model.state_dict(), os.path.join(output_dir, model_ckpt_file))
            torch.save(attn, os.path.join(output_dir, attention_matrix_file))

    model.load_state_dict(torch.load(os.path.join(output_dir, model_ckpt_file)))
    model.eval()

    with torch.no_grad():
        X_all = [torch.cat([train_data_orig[b], test_data[b]], dim=0).to(device)
                 for b in train_data if b in test_data]
        Y_all = [torch.cat([train_targets_orig[b], test_targets[b]], dim=0).cpu().numpy().tolist()
                 for b in train_data if b in test_targets]
        ids_all = [b for b in train_data if b in test_data]

        pred_all, attn = model(X_all)
        pred_seq = [p.cpu().numpy().tolist() for p in pred_all]
        save_predictions_per_basin_with_dates(folder, "all", ids_all, pred_seq, Y_all, seq_len)

    plot_loss(train_losses, test_losses)
    plot_attention(attn, ids_all)

    print(f"Total Basins: {len(ids_all)}")
    print(f"Attention Matrix Size: {attn.size()}")
    return model

# Main entry point for standalone runs
if __name__ == "__main__":
    from config import data_folder, basin_list_file
    import pandas as pd

    basin_list = pd.read_csv(basin_list_file, dtype=str)['basin'].tolist()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasinLevelCrossBasinAttention(
        input_dim=hparams['input_dim'],
        hidden_dim=hparams['hidden_dim'],
        num_layers=hparams['num_layers'],
        dropout=hparams['dropout'],
        context_dropout=hparams['context_dropout'],
        num_heads=hparams['num_heads']
    ).to(DEVICE)

    train_data, test_data, train_targets, test_targets = load_data(
        data_folder, basin_list, hparams['input_dim'], hparams['seq_length']
    )

    train_and_evaluate(
        model, train_data, train_targets,
        test_data, test_targets,
        DEVICE, folder=data_folder,
        seq_len=hparams['seq_length']
    )
