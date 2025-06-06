# tune.py
import os
import itertools
import pandas as pd
import torch
from model import BasinLevelCrossBasinAttention
from config import DEVICE, DATA_FOLDER, BASIN_LIST_FILE
from utils import load_data, shuffle_train_data
from train import train_and_evaluate  # Reusing your core train function

def get_hparam_grid():
    return {
        'hidden_dim': [128, 164],
        'dropout': [0.1, 0.3],
        'num_heads': [2, 4],
        'lr': [1e-4, 5e-4],
        'seq_length': [50, 100],
        'num_epochs': [100, 200],
        'context_dropout': [0.1, 0.3]
    }

def tune_hyperparams(fold_index):
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

        train_data, val_data, train_targets, val_targets = load_data(
            DATA_FOLDER, train_ids + val_ids, input_dim=55, seq_len=hparams['seq_length']
        )

        train_data = {k: v for k, v in train_data.items() if k in train_ids}
        val_data = {k: v for k, v in val_data.items() if k in val_ids}
        train_targets = {k: v for k, v in train_targets.items() if k in train_ids}
        val_targets = {k: v for k, v in val_targets.items() if k in val_ids}

        val_loss = train_and_evaluate(
            model=model,
            train_data=train_data,
            train_targets=train_targets,
            test_data=val_data,
            test_targets=val_targets,
            device=DEVICE,
            hparams=hparams,
            folder=DATA_FOLDER,
            seq_len=hparams['seq_length'],
            silent=True  # Avoids plots and saves
        )

        results.append({**hparams, "val_loss": val_loss})

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"output/tuning_results_fold{fold_index}.csv", index=False)

    best_row = results_df.loc[results_df["val_loss"].idxmin()]
    print(f"\nBest config for fold {fold_index}:")
    print(best_row.to_dict())

    return best_row.to_dict()

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    best_hparams = tune_hyperparams(fold_index=0)
