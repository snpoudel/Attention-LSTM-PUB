import os
import pandas as pd
import numpy as np
import time
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# config.py--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#
# Choose validation and test fold indices, rest will be used for training
TEST_FOLD_INDEX = 1
VALID_FOLD_INDEX = 0

# Hyperparameters
HPARAMS = {
    'input_dim': 36,
    'hidden_dim': 2, #256
    'num_layers': 1,
    'dropout': 0.4,
    'seq_length': 2, #365
    'num_epochs': 50, #max epochs
    'batch_size': 64,
    'lr': 1e-3
}

# Device
DEVICE = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

# Paths and filenames
DATA_FOLDER = "data/input"
BASIN_LIST_FILE = "data/basin_list_with_folds.csv"
OUTPUT_DIR = "output/vanilla_lstm"
FIGURE_DIR = "figures"
MODEL_CKPT_FILE = f"output/vanilla_lstm/{TEST_FOLD_INDEX}best_lstm_model.pt"
SCALER_FILE = "data/scaler.pkl"

# model.py--------------#----------------------#----------------------------#----------------------#--------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#
class VanillaLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim=1):
        super(VanillaLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_dim]
        last_hidden = self.dropout(lstm_out[:, -1, :])  # Get the last hidden state: [batch_size, hidden_dim]
        output = self.output_layer(last_hidden)  # output: [batch_size, output_dim]
        output = self.relu(output)
        return output.squeeze(-1)  # [batch_size] — prediction for each sequence
    
# Dataset Class--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#
class BasinSeqDataset(Dataset):
    def __init__(self, input_tensor, target_tensor, seq_len):
        self.inputs = input_tensor  # [n_basins, T, I]
        self.targets = target_tensor  # [n_basins, T]
        self.seq_len = seq_len

        self.n_basins, self.T, self.input_dim = self.inputs.shape
        self.indices = [(b, t) for b in range(self.n_basins) for t in range(self.T - seq_len)]

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        b, t = self.indices[idx]
        x = self.inputs[b, t:t+self.seq_len, :]  # [seq_len, input_dim]
        y = self.targets[b, t+self.seq_len-1]    # scalar
        return x, y

# Masked MSE Loss
def masked_loss(preds, targets):
    mask = ~torch.isnan(targets)
    return F.mse_loss(preds[mask], targets[mask])

# training script--------------#----------------------#----------------------------#----------------------#--------------#--------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#
start_time = time.time()

# Load basin info
basin_df = pd.read_csv(BASIN_LIST_FILE, dtype=str)
basin_df['fold'] = basin_df['fold'].astype(int)

valid_basins = basin_df[basin_df['fold'] == VALID_FOLD_INDEX]['basin'].tolist()
test_basins  = basin_df[basin_df['fold'] == TEST_FOLD_INDEX]['basin'].tolist()
train_basins = basin_df[~basin_df['basin'].isin(valid_basins + test_basins)]['basin'].tolist()

all_basins = train_basins + valid_basins
scaler = joblib.load(SCALER_FILE)

# Load data for all basins
input_list, target_list = [], []
for basin in all_basins:
    df = pd.read_csv(f"{DATA_FOLDER}/input_{basin}.csv")
    X = df.drop(columns=["Year", "Mnth", "Day", "qobs(mm/day)"]).values
    Y = df["qobs(mm/day)"].values
    input_list.append(scaler.transform(X))  # [T, I]
    target_list.append(Y)                   # [T]

input_tensor = torch.tensor(np.stack(input_list), dtype=torch.float32)    # [n, T, I]
target_tensor = torch.tensor(np.stack(target_list), dtype=torch.float32)  # [n, T]

# Split tensors into train and val sets
train_idx = [i for i, b in enumerate(all_basins) if b in train_basins]
valid_idx = [i for i, b in enumerate(all_basins) if b in valid_basins]

train_inputs = input_tensor[train_idx]
train_targets = target_tensor[train_idx]
valid_inputs = input_tensor[valid_idx]
valid_targets = target_tensor[valid_idx]

# Create datasets and loaders
seq_len = HPARAMS['seq_length']
batch_size = HPARAMS['batch_size']

train_dataset = BasinSeqDataset(train_inputs, train_targets, seq_len)
valid_dataset = BasinSeqDataset(valid_inputs, valid_targets, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# Initialize model
model = VanillaLSTM(
    input_dim=HPARAMS['input_dim'],
    hidden_dim=HPARAMS['hidden_dim'],
    num_layers=HPARAMS['num_layers'],
    dropout=HPARAMS['dropout']
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=HPARAMS['lr'])

# Training loop
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6) 
best_val_loss = float('inf')
patience_counter = 0
patience = 6

for epoch in range(1, HPARAMS['num_epochs'] + 1):
    model.train()
    train_losses = []

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x)  # [batch]
        loss = masked_loss(preds, y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    avg_train_loss = np.mean(train_losses)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = masked_loss(preds, y)
            if not torch.isnan(loss):
                val_losses.append(loss.item())
    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
    print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    # step up the scheduler
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

print(f"✅ Training completed in {(time.time() - start_time)/60:.2f} minutes. Best model saved to: {MODEL_CKPT_FILE}")

# Prediction script--------------#----------------------#----------------------------#----------------------#--------------#--------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#
start_time = time.time()
# Load test basin IDs
basin_df = pd.read_csv(BASIN_LIST_FILE, dtype=str)
basin_df['fold'] = basin_df['fold'].astype(int)
test_basins = basin_df[basin_df['fold'] == TEST_FOLD_INDEX]['basin'].tolist()

# Load scaler
scaler = joblib.load(SCALER_FILE)

# Model setup
model = VanillaLSTM(
    input_dim=HPARAMS['input_dim'],
    hidden_dim=HPARAMS['hidden_dim'],
    num_layers=HPARAMS['num_layers'],
    dropout=HPARAMS['dropout']
)
model.load_state_dict(torch.load(MODEL_CKPT_FILE, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

seq_len = HPARAMS['seq_length']

# Predict for each test basin
for basin_id in test_basins:
    df = pd.read_csv(f"{DATA_FOLDER}/input_{basin_id}.csv")
    df.rename(columns={'Mnth': 'Month'}, inplace=True)

    # Create date column
    date_col = pd.to_datetime(df[['Year', 'Month', 'Day']])

    X = df.drop(columns=["Year", "Month", "Day", "qobs(mm/day)"]).values
    Y = df["qobs(mm/day)"].values
    X_scaled = scaler.transform(X)

    T = len(Y)
    qsim = np.full(T, np.nan)

    with torch.no_grad():
        for t in range(T - seq_len + 1):
            x_seq = torch.tensor(X_scaled[t:t + seq_len], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            y_pred = model(x_seq).cpu().item()
            qsim[t + seq_len - 1] = y_pred

    df_out = pd.DataFrame({
        'Date': date_col,
        'qobs': Y,
        'qsim': qsim
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_out.to_csv(f"{OUTPUT_DIR}/predictions_{basin_id}.csv", index=False)
print(f"✅ All predictions completed in {(time.time() - start_time)/60:.2f} minutes.")
