# Attention-LSTM-PUB
A deep learning architecture that combines **LSTM with attention** to improve streamflow prediction in ungauged basins (PUB).

## Overview

`01space-time-attn-lstm.py` A complete single script that develops and implements the Attention-LSTM model for streamflow prediction in ungauged basins.

This repository contains python script that implements a deep learning Long Short-Term Memory (LSTM) model integrated with an attention mechanism specifically designed to improve streamflow predictions in ungauged basins. The code is setup for development in the 531 CAMELS-US basins, which are first divided into 7 folds, and in each run, 5 folds can be used for training, 1 fold for validation (early stopping), and 1 fold for out-of-sample in space testing (ungauged basin prediction). The model class is designed in a way that temporal and attention block are two modular components that can be toggled on or off, meaning a single model architecture can be used for four different configurations: (1) Only LSTM, (2) LSTM + Temporal Attention, (3) LSTM + Spatial Attention, and (4) LSTM + Temporal + Spatial Attention. The use of attention mechanism shows some promise to improve predictions and also shows potentially for some interpretability of model predictions, such as which time steps or which basins are more important for the prediction of a given basin.

This attention-based LSTM architecture for prediction in ungauged basins (PUB) is inspired by the idea that, while ungauged basins lack streamflow records, they still contain other hydrological input features that can be exploited during training to improve their streamflow predictions. By applying cross-basin attention at the output head of the LSTM model, ungauged basins can participate in the model training process by attending to basins that are similar to them. This enables ungauged basins to learn relationships from similar basins during training via attention weights, which are then leveraged to enhance their predictive capabilities.

# A related poster presentation 

![Poster Presentation at AGU 2025](AGU2025-Sandeep.pdf)
