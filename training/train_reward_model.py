import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib # To load the preprocessor if needed for feature count, though feature_names.json is better
from typing import List, Tuple # Added List, Tuple

class RewardModelMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=None, dropout_p=0.2):
        super(RewardModelMLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32] # Default hidden layer sizes

        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            current_size = hidden_size

        layers.append(nn.Linear(current_size, 1)) # Single output neuron for reward

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_data(data_dir: Path, batch_size: int) -> Tuple[DataLoader, DataLoader, int]:
    """Loads preprocessed training and validation data."""
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")

    feature_names_path = data_dir / "feature_names.json"
    input_size = X_train.shape[1]
    if feature_names_path.exists():
        try:
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
            # Validate that the number of feature names matches the number of columns in X_train
            if len(feature_names) == X_train.shape[1]:
                input_size = len(feature_names)
            else:
                print(f"Warning: Mismatch between feature_names.json ({len(feature_names)}) and X_train columns ({X_train.shape[1]}). Using X_train columns.")
        except Exception as e:
            print(f"Warning: Could not load or parse feature_names.json: {e}. Using X_train.shape[1] for input size.")

    print(f"Determined input_size for model: {input_size}")

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float().unsqueeze(1))
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float().unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, input_size

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path: Path):
    best_val_loss = float('inf')
    final_model_path = model_save_path / "reward_model_final.pth"
    best_model_path = model_save_path / "reward_model_best.pth"
    model_save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs} => Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to {best_model_path} (Val Loss: {best_val_loss:.4f})")

    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a Reward Model for RL.")
    parser.add_argument("--data_dir", type=str, default="../data/rm_training_data",
                        help="Directory containing preprocessed X_train.npy, y_train.npy, etc.")
    parser.add_argument("--model_save_dir", type=str, default="../models/reward_model",
                        help="Directory to save trained models.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--hidden_layers", type=str, default="128,64,32",
                        help="Comma-separated string of hidden layer sizes (e.g., '128,64,32').")
    parser.add_argument("--dropout_p", type=float, default=0.2, help="Dropout probability.")
    parser.add_argument("--no_cuda", action='store_true', help="Disable CUDA even if available.")

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    model_save_path = Path(args.model_save_dir)

    if not (data_path / "X_train.npy").exists() or not (data_path / "y_train.npy").exists():
        print(f"ERROR: Training data (X_train.npy or y_train.npy) not found at {data_path}. "
              "Please run `tools/prepare_rm_data.py` first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, input_size = load_data(data_path, args.batch_size)

    if input_size == 0 or len(train_loader.dataset) == 0: # Check if dataset is empty
        print("No data loaded or input size is zero. Cannot train model. Exiting.")
        return

    hidden_sizes_list = [int(s.strip()) for s in args.hidden_layers.split(',') if s.strip()] if args.hidden_layers else []

    model = RewardModelMLP(input_size, hidden_sizes=hidden_sizes_list, dropout_p=args.dropout_p).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting Reward Model training...")
    print(f"Model: {model}")
    print(f"Hyperparameters: Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}, HiddenLayers={hidden_sizes_list}, Dropout={args.dropout_p}")

    train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, device, model_save_path)
    print("Training complete.")

if __name__ == "__main__":
    main()
