'''
This script implements a Space-Time Attention LSTM model for out-of-sample in space (ungauged basins) streamflow prediction.
The model class is prepared such that spatial and temporal attention mechanisms can be toggled on or off, meaning
it can function as a plain LSTM, Temporal Attention LSTM, Spatial Attention LSTM, or Space-Time Attention LSTM.

Author: Sandeep Poudel
'''
import os
import pandas as pd
import numpy as np
import math
import time
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# config.py--------------#----------------------#--------------#----------------------#--------------#----------------------#
# 531 camels basins are divided in random 7 folds
# Choose 1 validation and 1 test fold indices, rest will be used for training
TEST_FOLD_INDEX = int(os.environ.get("test_fold", 0))  # Get from environment variable
VALID_FOLD_INDEX = (TEST_FOLD_INDEX + 1) % 7  # simple rotation rule, if test=0, valid=1, and rest is train

print(f"Using test fold: {TEST_FOLD_INDEX}, validation fold: {VALID_FOLD_INDEX}")
print(f"Running script: {os.path.basename(__file__)}")

# Hyperparameters
HPARAMS = {
    'input_dim': 36, #input features except for qobs
    'hidden_dim': 256, #256
    'num_layers': 1,
    'dropout': 0.4,
    'seq_length': 270, #270
    'num_epochs': 50, #50
    'lr': 1e-3,
}

# Device
DEVICE = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

# Paths and filenames
DATA_FOLDER = "data/input"
BASIN_LIST_FILE = "data/basin_list_with_folds.csv"
OUTPUT_DIR = "ensemble_output"
FIGURE_DIR = "figures"
MODEL_CKPT_FILE = f"ensemble_output/{TEST_FOLD_INDEX}best_model.pt"
SCALER_FILE = "data/scaler.pkl" # fitted StandardScaler from training data is already saved here

#model.py--------------#----------------------#----------------------#--------------#----------------------#--------------#----------------------#
class CrossBasinAttention(nn.Module):  # Space-Time Attention with LSTM
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.4, output_dim=1,
                 use_temporal=True, use_spatial=True):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=input_dim,              # I=40
                            hidden_size=hidden_dim,            # H=256
                            num_layers=num_layers,             # 1
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [n, T, I] — input for n basins, each with sequence of T and I features
        n, T, I = x.size()

        lstm_out, _ = self.lstm(x)  # [n, T, I] → [n, T, H]
        last_hidden = lstm_out[:, -1, :]  # [n, T, H] → [n, H]

        # Default context = last hidden state (plain LSTM)
        context = last_hidden
        temp_attn_weights, spat_attn_weights = None, None

        # Temporal attention
        if self.use_temporal:
            temp_attn_scores = torch.matmul(lstm_out, last_hidden.unsqueeze(2)) / math.sqrt(self.hidden_dim) # [n, T, H] @ [n, H, 1] → [n, T, 1]
            temp_attn_weights = torch.softmax(temp_attn_scores, dim=1)  # [n, T, 1]
            temp_out = torch.sum(temp_attn_weights * lstm_out, dim=1)    # [n, H]
            # Residual connection and layer normalization
            context = context + self.layernorm(temp_out)

        # Spatial attention
        if self.use_spatial:
            spat_attn_scores = torch.matmul(context, context.T) / math.sqrt(self.hidden_dim)
            spat_attn_weights = torch.softmax(spat_attn_scores, dim=1)  # [n, n]
            spat_out = torch.matmul(spat_attn_weights, context)          # [n, H]
            # Residual connection and layer normalization
            context = context + self.layernorm(spat_out)

        # Output layer
        output = self.fc(self.dropout(context))  # [n, H] → [n, 1]

        return output.squeeze(-1), {"spatial": spat_attn_weights, "temporal": temp_attn_weights}

# utils.py--------------#----------------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#
def load_data(folder, input_dim, seq_len, scaler_path, test_fold_idx, valid_fold_idx):
    basin_df = pd.read_csv(BASIN_LIST_FILE, dtype=str)
    basin_df['fold'] = basin_df['fold'].astype(int)

    valid_basins = basin_df[basin_df['fold'] == valid_fold_idx]['basin'].tolist()
    test_basins = basin_df[basin_df['fold'] == test_fold_idx]['basin'].tolist()
    train_basins = basin_df[~basin_df['basin'].isin(valid_basins + test_basins)]['basin'].tolist()
    all_basins = train_basins + valid_basins + test_basins
    n_basins = len(all_basins)

    # Load the fitted scaler
    scaler = joblib.load(scaler_path)

    input_seqs = []  # will become [n_basins, total_time, input_dim]
    target_seqs = [] # will become [n_basins, total_time]

    for basin in all_basins:
        df = pd.read_csv(f"{folder}/input_{basin}.csv")
        X = df.drop(columns=["Year", "Mnth", "Day", "qobs(mm/day)"]).values
        Y = df["qobs(mm/day)"].values

        X_scaled = scaler.transform(X)
        input_seqs.append(X_scaled)   # [T, I]
        target_seqs.append(Y)         # [T]

    # Stack across basins
    input_tensor = torch.tensor(np.stack(input_seqs), dtype=torch.float32)  # [n_basins, T, input_dim]
    target_tensor = torch.tensor(np.stack(target_seqs), dtype=torch.float32)  # [n_basins, T]

    # Identify split masks
    train_mask = torch.tensor([1 if b in train_basins else 0 for b in all_basins], dtype=torch.bool)  # [n_basins]
    valid_mask = torch.tensor([1 if b in valid_basins else 0 for b in all_basins], dtype=torch.bool)
    test_mask  = torch.tensor([1 if b in test_basins else 0 for b in all_basins], dtype=torch.bool)

    return input_tensor, target_tensor, train_mask, valid_mask, test_mask, all_basins

class BasinWindowDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, target_tensor, seq_len):
        self.inputs = input_tensor          # [n, T, I]
        self.targets = target_tensor        # [n, T]
        self.seq_len = seq_len
        self.n_basins, self.T, self.I = input_tensor.shape

        # only keep time indices where full window fits
        self.valid_time_indices = list(range(0, self.T - seq_len + 1))

    def __len__(self):
        return len(self.valid_time_indices)

    def __getitem__(self, idx):
        t_start = self.valid_time_indices[idx]
        t_end = t_start + self.seq_len

        x_window = self.inputs[:, t_start:t_end, :]         # [n, seq_len, I]
        y_window = self.targets[:, t_end - 1]               # [n]

        return x_window, y_window
        

# train.py--------------#----------------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#
# Timer
start_time = time.time()

# Load model
model = CrossBasinAttention(
    input_dim=HPARAMS['input_dim'],
    hidden_dim=HPARAMS['hidden_dim'],
    num_layers=HPARAMS['num_layers'],
    dropout=HPARAMS['dropout'],
).to(DEVICE)

# Load data
inputs, targets, train_mask, valid_mask, _, _ = load_data(
    folder=DATA_FOLDER,
    input_dim=HPARAMS['input_dim'],
    seq_len=HPARAMS['seq_length'],
    scaler_path=SCALER_FILE,
    test_fold_idx=TEST_FOLD_INDEX,
    valid_fold_idx=VALID_FOLD_INDEX
)

# Train dataset
dataset = BasinWindowDataset(inputs, targets, seq_len=HPARAMS['seq_length'])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
valid_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Optimizer and loss
def namask_loss_fn(x, y):
    mask = ~torch.isnan(y)
    return F.mse_loss(x[mask], y[mask])

loss_fn = namask_loss_fn
optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS['lr'])
# Training loop
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-6) # scheduler for learning rate decay
best_val_loss = float('inf')
patience = 5

for epoch in range(HPARAMS['num_epochs']):
    model.train()
    epoch_loss = 0.0
    for xb, yb in dataloader:
        xb = xb.squeeze(0).to(DEVICE)  # [n, seq_len, I]
        yb = yb.squeeze(0).to(DEVICE)  # [n]

        y_pred, _ = model(xb)  # [n]
        loss = loss_fn(y_pred[train_mask], yb[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(dataloader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb = xb.squeeze(0).to(DEVICE)
            yb = yb.squeeze(0).to(DEVICE)  # [n]
            y_pred, _ = model(xb)  # [n]
            loss = loss_fn(y_pred[valid_mask], yb[valid_mask])
            val_loss += loss.item()
    avg_val_loss = val_loss / len(valid_loader)

    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Time: {time.time() - start_time:.1f}s")
    # step up scheduler for learning rate decay
    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss - 1e-4:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_CKPT_FILE)
        print(f"✔️ Best model saved at epoch {epoch}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break
print(f"Training completed in {(time.time() - start_time)/60:.1f} minutes")

# predict.py--------------#----------------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#
start_time = time.time()
# === Load trained model ===
model = CrossBasinAttention(
    input_dim=HPARAMS['input_dim'],
    hidden_dim=HPARAMS['hidden_dim'],
    num_layers=HPARAMS['num_layers'],
    dropout=HPARAMS['dropout'],
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_CKPT_FILE,  weights_only=True))

model.eval()
# === Load data ===
input_tensor, target_tensor, train_mask, valid_mask, test_mask, all_basins = load_data(
    folder=DATA_FOLDER,
    input_dim=HPARAMS['input_dim'],
    seq_len=HPARAMS['seq_length'],
    scaler_path=SCALER_FILE,
    test_fold_idx=TEST_FOLD_INDEX,
    valid_fold_idx=VALID_FOLD_INDEX
)

dates_dict = {}
for basin in all_basins:
    df = pd.read_csv(f"{DATA_FOLDER}/input_{basin}.csv")
    df = df.rename(columns={'Year': 'year', 'Mnth': 'month', 'Day': 'day'})
    dates_dict[basin] = pd.to_datetime(df[['year', 'month', 'day']])

# === Set up dataset and dataloader ===
dataset = BasinWindowDataset(input_tensor, target_tensor, seq_len=HPARAMS['seq_length'])
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# === Make predictions ===
n_basins, T, _ = input_tensor.shape
seq_len = HPARAMS['seq_length']
preds_per_basin = [[] for _ in range(n_basins)]

with torch.no_grad():
    for xb, _ in dataloader:
        xb = xb.squeeze(0).to(DEVICE)  # [n, seq_len, input_dim]
        y_pred, _ = model(xb)          # [n]
        for i in range(n_basins):
            preds_per_basin[i].append(y_pred[i].item())

# === Save predictions per basin ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_indices = torch.where(test_mask)[0].tolist()
# test_indices = list(range(n_basins))  # Save for all basins

for i in test_indices:
    basin = all_basins[i]
    dates = dates_dict[basin][seq_len - 1:]
    obs = target_tensor[i, seq_len - 1:].numpy()
    sim = np.array(preds_per_basin[i])

    df = pd.DataFrame({
        'date': dates.values[:len(sim)],
        'obs': np.round(obs[:len(sim)], 4),
        'sim': np.round(np.clip(sim, 0, None), 4)  # Ensure non-negative predictions and round to 4 decimals
    })

    out_path = os.path.join(OUTPUT_DIR, f"predictions_{basin}.csv")
    df.to_csv(out_path, index=False)
print(f'✅ Prediction completed in {(time.time() - start_time)/60:.2f} minutes')