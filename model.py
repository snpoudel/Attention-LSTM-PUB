# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossBasinAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_heads, context_dropout):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.context_dropout = context_dropout

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # self.attn_temp = nn.Parameter(torch.tensor(0.5))
        #use log temperature for more stable training
        self.log_attn_temp = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_basin):
        all_last_hidden = []
        basin_reps = []

        for seq_batch in x_basin: #seq_batch is all sequences for a basin
            lstm_out, _ = self.lstm(seq_batch)
            last_hidden = lstm_out[:, -1, :] # last hidden state for each sequence in the batch
            basin_reps.append(last_hidden.mean(dim=0)) # mean across all sequences in the basin to get a basin representation
            all_last_hidden.append(last_hidden)

        H_b = torch.stack(basin_reps)
        B = H_b.size(0)

        Q = F.normalize(self.W_q(H_b).view(B, self.num_heads, self.head_dim), dim=2) #reshape mor multi-head and normalize across heads
        K = F.normalize(self.W_k(H_b).view(B, self.num_heads, self.head_dim), dim=2)
        V = self.W_v(H_b).view(B, self.num_heads, self.head_dim)

        # attn_scores = torch.einsum("bnd,cnd->bnc", Q, K) / torch.clamp(self.attn_temp, min=1e-2)
        temperature = torch.exp(self.log_attn_temp) + 1e-2  # Ensure temperature is positive
        attn_scores = torch.einsum("bnd,cnd->bnc", Q, K) / temperature
        attn_weights = F.softmax(attn_scores, dim=-1)

        context = torch.einsum("bnc,cnd->bnd", attn_weights, V)
        context = context.reshape(B, -1)
        context_out = self.out_proj(context)

        seq_preds = []
        for i in range(B):
            local = all_last_hidden[i]
            global_ = context_out[i].unsqueeze(0).expand_as(local)

            if self.training and torch.rand(1).item() < self.context_dropout:
                combined = torch.cat([torch.zeros_like(local), global_], dim=1)
            else:
                combined = torch.cat([local, global_], dim=1)

            y = self.decoder(combined).squeeze(-1)
            seq_preds.append(y)

        return seq_preds, attn_weights.mean(dim=1)


class VanillaLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_basin):
        all_last_hidden = []
        for seq_batch in x_basin: #seq_batch is all sequences for a basin
            lstm_out, _ = self.lstm(seq_batch)
            last_hidden = lstm_out[:, -1, :] # last hidden state for each sequence in the batch
            preds = self.decoder(last_hidden).squeeze(-1)
            all_last_hidden.append(preds)
        return all_last_hidden