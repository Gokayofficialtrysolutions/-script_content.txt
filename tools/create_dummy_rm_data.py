import numpy as np
import json
from pathlib import Path
import os # Added for path manipulation if needed, though Path should suffice

def generate_dummy_data(output_dir_str: str, num_train: int, num_val: int, num_features: int):
    # Ensure output_dir_str is treated as relative to the script's execution path or make it absolute
    # For simplicity, assuming it's relative to where the script is called from (e.g. repo root)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.random.rand(num_train, num_features).astype(np.float32)
    y_train = np.random.rand(num_train).astype(np.float32)
    X_val = np.random.rand(num_val, num_features).astype(np.float32)
    y_val = np.random.rand(num_val).astype(np.float32)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)

    feature_names = [f'feature_{i}' for i in range(num_features)]
    with open(output_dir / "feature_names.json", 'w') as f:
        json.dump(feature_names, f)

    print(f"Dummy data created in {output_dir}")

if __name__ == "__main__":
    # Default parameters for dummy data generation
    # The paths used by train_reward_model.py are relative to its own location (../data/rm_training_data)
    # So, this script should create data in a location accessible by that relative path from training/
    # If this script is in tools/ and train_reward_model.py is in training/,
    # and they both expect data in data/rm_training_data relative to repo root,
    # then "data/rm_training_data" is correct.
    data_path_str = "data/rm_training_data"
    generate_dummy_data(data_path_str, num_train=100, num_val=20, num_features=10)
