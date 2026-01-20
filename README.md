# Attention-LSTM-PUB

An attention-based LSTM architecture for improving **streamflow prediction in ungauged basins (PUB)**.

## Overview

This repository contains a single, self-contained Python script:

- **`01space-time-attn-lstm.py`**  
  Implements an LSTM model with optional **temporal** and **spatial (cross-basin)** attention mechanisms for PUB prediction.

The model is developed using **531 CAMELS-US basins**, split into **7 spatial folds**:
- 5 folds for training  
- 1 fold for validation (early stopping)  
- 1 fold for out-of-sample spatial testing (ungauged basins)

## Model Configurations

A single modular architecture supports four configurations:
1. LSTM only  
2. LSTM + Temporal Attention  
3. LSTM + Spatial Attention  
4. LSTM + Temporal + Spatial Attention  


## Motivation

This attention-based LSTM architecture for prediction in ungauged basins (PUB) is motivated by a key idea: while ungauged basins lack streamflow records, they still contain valuable hydrological input features that can be directly leveraged during training to improve streamflow predictions. 

The spatial attention mechanism operates across basins at the LSTM's output head/hidden states, allowing ungauged basins to actively participate in model training by attending to similar basins. Through learned attention weights, ungauged basins can capture hydrological relationships from basins with comparable characteristics, enhancing their predictive capabilities even without direct streamflow observations during training.

The temporal attention mechanism addresses a limitation of standard LSTMs, which rely solely on the final hidden state for predictions. By learning to focus on the most relevant time steps within the input sequence, temporal attention enables the model to identify and weight important patterns across the lookback period, potentially improving prediction accuracy.

Together, the space-time attention mechanism provides the model with additional flexibility to directly learn from both spatially similar basins and temporally relevant inputs. When making a prediction for a given basin at a specific time step, the architecture can attend to all training basins across the entire lookback period (1 year), effectively learning which basin-time combinations are most informative for the prediction task.


## Related Presentation

Poster presented at **AGU 2025** outlining more details and results from this work:

![Poster Presentation at AGU 2025](AGU2025-Sandeep.png)