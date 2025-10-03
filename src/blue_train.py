import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from red_datasetloader import AdversarialDataloader
from model import SimpleCNN

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch", type=int, default=64, help="Batch size")
parser.add_argument(
    "--save",
    type=str,
    default="../models/model_adversarial.pt",
    help="Path to save the trained model",
)
parser.add_argument(
    "--attack_type",
    type=str,
    default="fgsm",
    choices=["fgsm", "pgd"],
    help="Attack type for adversarial samples",
)
parser.add_argument(
    "--epsilon", type=float, default=0.1, help="Epsilon value for adversarial attack"
)
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load adversarial and clean data
adversarial_dataloader = AdversarialDataloader.configured_dataloader()
from model import SimpleCNN

model = SimpleCNN()
model.load_state_dict(
    torch.load("../models/model_mnist.pt", map_location="cuda", weights_only=True)
)
model.eval()
adversarial_dataloader.model = model
adversarial_dataloader.attack_type = args.attack_type
adversarial_dataloader.epsilon = args.epsilon
(x_train_adv, y_train_adv), (x_test, y_test) = adversarial_dataloader.load_data()

# Combine clean and adversarial data for training
x_train = np.concatenate((x_train_adv, x_train_adv), axis=0)
y_train = np.concatenate((y_train_adv, y_train_adv), axis=0)

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
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item() * X.size(0)
            val_samples += X.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

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

# Evaluate the model
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Calculate accuracy
accuracy = (all_preds == all_labels).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate confusion matrix and classification report
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds, digits=4)

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

# Save evaluation results
report_filename = f"../models/eval_results_adversarial_{args.attack_type}.txt"
with open(report_filename, "w") as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix, separator=", ") + "\n\n")
    f.write("Classification Report:\n")
    f.write(class_report)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10),
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title(f"Confusion Matrix (Adversarial + Clean Training)")
plt.savefig(f"../models/confusion_matrix_adversarial_{args.attack_type}.png")
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, args.epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss (Adversarial + Clean Training)")
plt.legend()
plt.grid()
plt.savefig(f"../models/training_plot_adversarial_{args.attack_type}.png")
plt.show()

print("Training and evaluation complete. Results saved to '../models/'.")
