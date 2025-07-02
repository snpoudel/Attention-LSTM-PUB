# config.py

import torch

# Hyperparameters
hparams = {
    'input_dim': 55,
    'hidden_dim': 164,
    'num_layers': 1,
    'dropout': 0.3,
    # 'context_dropout': 0.5,
    # 'num_heads': 4,
    'seq_length': 100,
    'num_epochs': 800,
    'lr': 1e-4
}
#test fold index
#use 0 for tuning, and 1 or 2 or 3 or 4 for training
TEST_FOLD_INDEX = 1

# Device
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# Paths and filenames (now in UPPERCASE)
DATA_FOLDER = "data/input"
BASIN_LIST_FILE = "data/basin_list_with_folds.csv"
OUTPUT_DIR = "output/vanilla_lstm"
FIGURE_DIR = "figures"
MODEL_CKPT_FILE = f"{TEST_FOLD_INDEX}best_lstm_model.pt"
ATTENTION_MATRIX_FILE = f"{TEST_FOLD_INDEX}best_lstm_attention.pt"

