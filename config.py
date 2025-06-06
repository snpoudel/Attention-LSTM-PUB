# config.py

import torch

# Hyperparameters
hparams = {
    'input_dim': 55,
    'hidden_dim': 4,
    'num_layers': 1,
    'dropout': 0.3,
    'context_dropout': 0.3,
    'num_heads': 4,
    'seq_length': 2,
    'num_epochs': 5,
    'lr': 1e-4
}

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths and filenames (now in UPPERCASE)
DATA_FOLDER = "data/input"
BASIN_LIST_FILE = "data/basin_list_with_folds.csv"
OUTPUT_DIR = "output"
FIGURE_DIR = "figures"
MODEL_CKPT_FILE = "best_model.pt"
ATTENTION_MATRIX_FILE = "best_attention.pt"
