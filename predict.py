'''This script is not used now, as the train script itself haldles prediction for entire datset now
However, in future, if we want to run prediction in new dataset such as under climate change, we might need this script'''

# # predict.py

# import os
# import torch
# import pandas as pd
# import numpy as np
# from config import hparams, data_folder, output_dir, model_ckpt_file
# from model import BasinLevelCrossBasinAttention
# from utils import load_data, save_predictions_per_basin_with_dates

# def run_prediction():
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     basin_list = pd.read_csv("basin_list.csv", dtype=str)['basin'].tolist()

#     # Load data
#     train_data, test_data, train_targets, test_targets = load_data(
#         data_folder, basin_list, hparams['input_dim'], hparams['seq_length']
#     )

#     # Init model and load checkpoint
#     model = BasinLevelCrossBasinAttention(
#         input_dim=hparams['input_dim'],
#         hidden_dim=hparams['hidden_dim'],
#         num_layers=hparams['num_layers'],
#         dropout=hparams['dropout'],
#         context_dropout=hparams['context_dropout'],
#         num_heads=hparams['num_heads']
#     ).to(DEVICE)
#     model.load_state_dict(torch.load(os.path.join(output_dir, model_ckpt_file)))
#     model.eval()

#     with torch.no_grad():
#         X_all = [torch.cat([train_data[b], test_data[b]], dim=0).to(DEVICE)
#                  for b in basin_list if b in train_data and b in test_data]
#         Y_all = [torch.cat([train_targets[b], test_targets[b]], dim=0).cpu().numpy().tolist()
#                  for b in basin_list if b in train_targets and b in test_targets]
#         pred_all, _ = model(X_all)
#         pred_seq = [p.cpu().numpy().tolist() for p in pred_all]
#         save_predictions_per_basin_with_dates(data_folder, "predict", basin_list, pred_seq, Y_all, hparams['seq_length'])

#     print("✅ Prediction completed and saved!")

# if __name__ == "__main__":
#     run_prediction()
