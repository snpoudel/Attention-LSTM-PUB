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
    'input_dim': 36, #input features except for qobs
    'hidden_dim': 2, #256
    'num_layers': 1,
    'dropout': 0.4,
    'num_heads': 1,
    'seq_length': 2, #356
    'num_epochs': 1,
    'lr': 1e-3,
}

# Device
DEVICE = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

# Paths and filenames
DATA_FOLDER = "data/input"
BASIN_LIST_FILE = "data/basin_list_with_folds.csv"
OUTPUT_DIR = "output"
FIGURE_DIR = "figures"
MODEL_CKPT_FILE = f"output/{TEST_FOLD_INDEX}best_model.pt"
ATTENTION_MATRIX_FILE = f'attn_sample.npy'
SCALER_FILE = "data/scaler.pkl"

#model.py--------------#----------------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#
class CrossBasinAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_heads):
        """
        Input:
            x: [n, T, I] where
                n = number of basins (batch size),
                T = sequence length,
                I = input dimension per time step.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)  # Input: [n, T, I] → Output: [n, T, H]
        # Attention projection layers
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.log_attn_temp = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # MLP decoder: [n, 2H] → [n, 1]
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()  # Ensure output is non-negative
        )
    def forward(self, x):  
        """
        x: [n, T, I] → n basins, each with a sequence of T steps of input_dim=I
        Output:
            preds: [n] — 1 prediction per basin
            attn_weights_mean: [n] — average attention received by each basin
        """
        n = x.size(0)  # number of basins
        # LSTM encoding
        lstm_out, _ = self.lstm(x)               # [n, T, H]
        h_n = lstm_out[:, -1, :]                 # last hidden state: [n, H]
        # Multi-head attention over basins
        Q = F.normalize(self.W_q(h_n).view(n, self.num_heads, self.head_dim), dim=2)  # [n, heads, head_dim]
        K = F.normalize(self.W_k(h_n).view(n, self.num_heads, self.head_dim), dim=2)  # [n, heads, head_dim]
        V = self.W_v(h_n).view(n, self.num_heads, self.head_dim)                     # [n, heads, head_dim]

        temperature = torch.exp(self.log_attn_temp) + 1e-2
        attn_scores = torch.einsum("bhd,chd->bhc", Q, K) / temperature               # [n, heads, n]
        attn_weights = F.softmax(attn_scores, dim=-1)                                # [n, heads, n]
        context = torch.einsum("bhc,chd->bhd", attn_weights, V)                      # [n, heads, head_dim]
        context = context.reshape(n, -1)                                             # [n, H]
        global_context = self.out_proj(context)                                         # [n, H]
        # Combine local (h_n) and global (context_out) after dropout of local context
        if self.training:
            h_n = F.dropout(h_n, p=self.dropout, training=True)             # [n, H]
        local_context = h_n
        combined = torch.cat([local_context, global_context], dim=1)  # [n, 2H]
        preds = self.decoder(combined).squeeze(-1)                                   # [n]
        attn_weights_mean = attn_weights.mean(dim=1)                                 # [n] — average over heads
        return preds, attn_weights_mean
    
# utils.py--------------#----------------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#
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

    return input_tensor, target_tensor, train_mask, valid_mask, test_mask

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
        

# train.py--------------#----------------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#
# Timer
start_time = time.time()

# Load model
model = CrossBasinAttention(
    input_dim=HPARAMS['input_dim'],
    hidden_dim=HPARAMS['hidden_dim'],
    num_layers=HPARAMS['num_layers'],
    dropout=HPARAMS['dropout'],
    # context_dropout=HPARAMS['context_dropout'],
    num_heads=HPARAMS['num_heads']
).to(DEVICE)

# Load data
inputs, targets, train_mask, valid_mask, _ = load_data(
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

# Optimizer and loss
def namask_loss_fn(x, y):
    mask = ~torch.isnan(y)
    return F.mse_loss(x[mask], y[mask])
loss_fn = namask_loss_fn

optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS['lr'])

# Training loop
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-7) # scheduler for learning rate decay
best_val_loss = float('inf')
patience = 6
epochs_no_improve = 0

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
    val_losses = []
    with torch.no_grad():
        for t in range(0, inputs.size(1) - HPARAMS['seq_length'] + 1):
            xval = inputs[:, t:t+HPARAMS['seq_length'], :].to(DEVICE)
            yval = targets[:, t+HPARAMS['seq_length']-1].to(DEVICE)

            ypred, _ = model(xval)
            loss = loss_fn(ypred[valid_mask], yval[valid_mask])
            val_losses.append(loss.item())

    avg_val_loss = np.mean(val_losses)

    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Temp: {torch.exp(model.log_attn_temp).item():.3f} | "
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

# predict.py--------------#----------------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#--------------#----------------------#
start_time = time.time()
# === Load trained model ===
model = CrossBasinAttention(
    input_dim=HPARAMS['input_dim'],
    hidden_dim=HPARAMS['hidden_dim'],
    num_layers=HPARAMS['num_layers'],
    dropout=HPARAMS['dropout'],
    # context_dropout=HPARAMS['context_dropout'],
    num_heads=HPARAMS['num_heads']
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_CKPT_FILE,  weights_only=True))
model.eval()

# === Load data ===
input_tensor, target_tensor, train_mask, valid_mask, test_mask = load_data(
    folder=DATA_FOLDER,
    input_dim=HPARAMS['input_dim'],
    seq_len=HPARAMS['seq_length'],
    scaler_path=SCALER_FILE,
    test_fold_idx=TEST_FOLD_INDEX,
    valid_fold_idx=VALID_FOLD_INDEX
)

# === Get basin info and dates ===
basin_df = pd.read_csv(BASIN_LIST_FILE, dtype=str)
all_basins = basin_df['basin'].tolist()
test_basins = basin_df[basin_df['fold'] == str(TEST_FOLD_INDEX)]['basin'].tolist()

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

test_indices = [all_basins.index(b) for b in test_basins]

for i in test_indices:
    basin = all_basins[i]
    dates = dates_dict[basin][seq_len - 1:]
    obs = target_tensor[i, seq_len - 1:].numpy()
    sim = np.array(preds_per_basin[i])

    df = pd.DataFrame({
        'date': dates.values[:len(sim)],
        'obs': obs[:len(sim)],
        'sim': np.round(sim, 3)
    })

    out_path = os.path.join(OUTPUT_DIR, f"predictions_{basin}.csv")
    df.to_csv(out_path, index=False)
print(f'✅ Prediction completed in {(time.time() - start_time)/60:.2f} minutes')