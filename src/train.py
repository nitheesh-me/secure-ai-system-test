import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasetloader import MnistDataloader
from model import SimpleCNN

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch", type=int, default=64, help="Batch size")
parser.add_argument(
    "--save",
    type=str,
    default="../models/model_mnist.pt",
    help="Path to save the trained model",
)
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
mnist_dataloader = MnistDataloader.configured_dataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Convert data to PyTorch tensors
x_train = torch.tensor(np.array(x_train), dtype=torch.float32).unsqueeze(1) / 255.0
y_train = torch.tensor(np.array(y_train), dtype=torch.long)
x_test = torch.tensor(np.array(x_test), dtype=torch.float32).unsqueeze(1) / 255.0
y_test = torch.tensor(np.array(y_test), dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

val_dataset = TensorDataset(x_test, y_test)
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
train_losses = []
val_losses = []
best_val_loss = float("inf")

for epoch in range(1, args.epochs + 1):
    model.train()
    running_loss = 0.0
    total_samples = 0
    start_time = time.time()

    # Training with tqdm progress bar
    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch}/{args.epochs}")
        for X, y in tepoch:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            total_samples += X.size(0)
            tepoch.set_postfix(loss=loss.item())

    train_loss = running_loss / total_samples
    train_losses.append(train_loss)

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_samples = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item() * X.size(0)
            val_samples += X.size(0)

    val_loss /= val_samples
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {time.time() - start_time:.2f}s"
    )

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), args.save)
        print(f"Best model saved to {args.save}")

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, args.epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.savefig("../models/training_plot.png")
plt.show()

print("Training complete. Loss plot saved as 'training_plot.png'.")
