import pandas as pd
import numpy as np

def assign_folds(input_csv="data/basin_list.csv", output_csv="data/basin_list_with_folds.csv", n_folds=5, seed=42):
    df = pd.read_csv(input_csv, dtype=str)
    np.random.seed(seed)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    shuffled["fold"] = np.tile(np.arange(n_folds), int(np.ceil(len(df) / n_folds)))[:len(df)]
    shuffled.to_csv(output_csv, index=False)
    print(f"Assigned folds to {len(df)} basins and saved to {output_csv}")

if __name__ == "__main__":
    assign_folds()

# df = pd.read_csv("basin_list_with_folds.csv", dtype=str)
# train_ids = df[df["fold"] != fold_idx]["basin"].tolist()
# test_ids = df[df["fold"] == fold_idx]["basin"].tolist()

