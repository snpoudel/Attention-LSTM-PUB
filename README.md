# LSTM-Attention4PUB
A deep learning architecture that combines **LSTM with cross-basin attention** to improve streamflow prediction in ungauged basins (PUB).

## Repo Structure
```
LSTM-Attention4PUB/
├── config.py              # Global configuration (device, paths, seed, hyperparameters)
├── model.py               # Cross-Basin Attention LSTM model definition
├── train.py               # Train the model on specified folds
├── predict.py             # Generate predictions using the trained model
├── plot.py                # Plot loss curves, attention heatmaps, and evaluation metrics
├── utils.py               # Data loading, preprocessing, and utility functions
├── tune.py                # One-fold hyperparameter tuning script
├── assign_folds.py        # Script to assign 5-fold split to basins
├── data/
│   └── lstm_input_<basin>.csv   # Per-basin standardized time series data
├── output/
│   └── <predictions>.csv        # Prediction outputs saved per basin
├── figures/
│   ├── <loss curves>.png
│   └── <attention heatmaps>.png
└── README.md              # This file

```

## How to Use this Repo:
1. Install dependencies using `pip install -r requirements.txt`.
2. Run `assign_folds.py` to assign each basin to one of the 5 folds.
3. Run `tune.py` to perform hyperparameter tuning on a single fold. Then update the best found hyperparameters in `config.py`.
4. Train the model using `train.py` with the desired fold.
5. Run predictions using `predict.py` on the trained model.
6. Visualize results using `plot.py` to generate loss curves, attention matrices, and evaluation metrics.

PS: The Cross-Basin Attention LSTM model developed in this project is inside `model.py`. The utility functions for data loading, randomizing sequences, saving predictions, and making plots are in `utils.py`.

## Additional Notes
The deep learning-based **LSTM with cross-basin attention** architecture for prediction in ungauged basins (PUB) is inspired by the idea that, while ungauged basins lack streamflow records, they still contain other hydrological input features that can be exploited during training to improve their streamflow predictions. By applying cross-basin attention at the output head of the LSTM model, ungauged basins can participate in the model training process by attending to basins that are similar to them. This enables ungauged basins to learn relationships from similar basins during training via attention weights, which are then leveraged to enhance their predictive capabilities.