import os
import itertools
import pandas as pd
import torch
import time
from model import BasinLevelCrossBasinAttention
from config import DEVICE, DATA_FOLDER, BASIN_LIST_FILE
from utils import load_data, shuffle_train_data

start_time = time.time()

def get_hparam_grid():
    return {
        'hidden_dim': [4],
        'dropout': [0.3],
        'num_heads': [4],
        'lr': [1e-4],
        'seq_length': [2],
        'num_epochs': [1],
        'context_dropout': [0.2,0.3]
    }

def train_and_evaluate(model, all_data, train_targets, val_targets,
                       device, folder, seq_len, lr, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        shuffled_data, shuffled_targets = shuffle_train_data(all_data, train_targets)
        X_b = [shuffled_data[b].to(device) for b in shuffled_data]
        Y_b = [shuffled_targets[b].to(device) for b in shuffled_targets]

        pred, _ = model(X_b)
        mask = ~torch.isnan(torch.cat(Y_b))
        loss = loss_fn(torch.cat(pred)[mask], torch.cat(Y_b)[mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            X_val = [all_data[b].to(device) for b in all_data]
            Y_val = [val_targets[b].to(device) for b in val_targets]
            pred_val, _ = model(X_val)
            mask_val = ~torch.isnan(torch.cat(Y_val))
            val_loss = loss_fn(torch.cat(pred_val)[mask_val], torch.cat(Y_val)[mask_val])
            val_losses.append(val_loss.item())

    return val_losses[-1]  # or min(val_losses)

def tune_hyperparams(fold_index=0):
    basin_df = pd.read_csv(BASIN_LIST_FILE, dtype=str)
    train_ids = basin_df[basin_df["fold"] != str(fold_index)]["basin"].tolist()
    val_ids = basin_df[basin_df["fold"] == str(fold_index)]["basin"].tolist()

    print(f"Tuning on fold {fold_index}: {len(train_ids)} train basins, {len(val_ids)} val basins")

    grid = get_hparam_grid()
    keys, values = zip(*grid.items())
    combinations = list(itertools.product(*values))

    results = []

    for i, comb in enumerate(combinations):
        hparams = dict(zip(keys, comb))
        print(f"\n[{i+1}/{len(combinations)}] Trying config: {hparams}")

        model = BasinLevelCrossBasinAttention(
            input_dim=55,
            hidden_dim=hparams['hidden_dim'],
            num_layers=1,
            dropout=hparams['dropout'],
            num_heads=hparams['num_heads'],
            context_dropout=hparams['context_dropout']
        ).to(DEVICE)

        # Load full input data (all basins), with masking handled internally
        all_data, train_targets, val_targets = load_data(
            DATA_FOLDER, input_dim=55, seq_len=hparams['seq_length'], test_fold_idx=fold_index
        )

        val_loss = train_and_evaluate(
            model=model,
            all_data=all_data,
            train_targets=train_targets,
            val_targets=val_targets,
            device=DEVICE,
            folder=DATA_FOLDER,
            seq_len=hparams['seq_length'],
            lr=hparams['lr'],
            num_epochs=hparams['num_epochs']
        )

        results.append({**hparams, "val_loss": val_loss})

    results_df = pd.DataFrame(results)
    os.makedirs("output", exist_ok=True)
    results_df.to_csv(f"output/best_tuning_results_fold{fold_index}.csv", index=False)

    best_row = results_df.loc[results_df["val_loss"].idxmin()]
    print(f"\nBest config for fold {fold_index}:")
    print(best_row.to_dict())

    return best_row.to_dict()

if __name__ == "__main__":
    best_hparams = tune_hyperparams(fold_index=0)

end_time = time.time()
print(f'Completed total hyperparam combinations: {len(best_hparams)} in {end_time - start_time:.2f} seconds')