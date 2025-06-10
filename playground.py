import os
import pandas as pd

def process_csv_files(input_folder, output_folder, decimal_places=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in (f for f in os.listdir(input_folder) if f.endswith('.csv')):
        df = pd.read_csv(os.path.join(input_folder, file_name)).round(decimal_places)
        df.to_csv(os.path.join(output_folder, file_name), index=False)

if __name__ == "__main__":
    input_folder = 'data/old_input/'
    output_folder = 'data/input/'
    process_csv_files(input_folder, output_folder)
    print(f"Rounded CSV files saved to {output_folder}")
