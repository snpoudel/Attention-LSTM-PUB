# config.py

# Hyperparameters
hparams = {
    'input_dim': 55,
    'hidden_dim': 164,
    'num_layers': 1,
    'dropout': 0.3,
    'context_dropout': 0.3,  #dropout of local context in the LSTM-attention model
    'num_heads': 4,
    'seq_length': 100,
    'num_epochs': 600,
    'lr': 1e-4
}

# Paths and folder names
data_folder = "pub_lstm_input"
basin_list_file = "basin_list.csv"
output_dir = "output"
figure_dir = "figures"
model_ckpt_file = "best_model.pt"
attention_matrix_file = "best_attention.pt"

# Cross-validation
n_folds = 5
