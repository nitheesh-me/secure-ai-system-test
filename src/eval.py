import torch
import time
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasetloader import MnistDataloader
from red_datasetloader import PoisonedDataloader, AdversarialDataloader
from model import SimpleCNN

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="../models/model_adversarial.pt", help="Path to the trained model"
)
parser.add_argument("--batch", type=int, default=256, help="Batch size for evaluation")
parser.add_argument(
    "--dataloader",
    type=str,
    default="mnist",
    choices=["mnist", "poisoned", "adversarial"],
    help="Type of dataloader to use",
)
parser.add_argument(
    "--attack_type",
    type=str,
    default="fgsm",
    choices=["fgsm", "pgd"],
    help="Attack type for adversarial dataloader",
)
parser.add_argument(
    "--epsilon", type=float, default=0.1, help="Epsilon value for adversarial attack"
)
args = parser.parse_args()

if args.dataloader == "adversarial":
    prefix = f"{args.dataloader}_{args.attack_type}"
else:
    prefix = args.dataloader

report_filename = f"../models/f_eval_results_{prefix}.txt"
conf_matrix_filename = f"../models/f_confusion_matrix_{prefix}.png"
class_metrics_filename = f"../models/f_class_metrics_{prefix}.png"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configure the dataloader
if args.dataloader == "mnist":
    dataloader = MnistDataloader.configured_dataloader()
elif args.dataloader == "poisoned":
    dataloader = PoisonedDataloader.configured_dataloader()
elif args.dataloader == "adversarial":
    # Load the pre-trained model for adversarial dataloader
    model = SimpleCNN().to(device)
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    dataloader = AdversarialDataloader.configured_dataloader()
    dataloader.model = model
    dataloader.attack_type = args.attack_type
    dataloader.epsilon = args.epsilon
else:
    raise ValueError("Invalid dataloader type specified.")

# Load the test dataset
(_, _), (x_test, y_test) = dataloader.load_data()

# Convert data to PyTorch tensors
x_test = torch.tensor(np.array(x_test), dtype=torch.float32).unsqueeze(1) / 255.0
y_test = torch.tensor(np.array(y_test), dtype=torch.long)

val_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

# Load the trained model
model = SimpleCNN().to(device)
if not os.path.exists(args.model):
    raise FileNotFoundError(f"Model file not found: {args.model}")
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()
print(f"Loaded model from {args.model}")

# Evaluate the model
all_preds, all_labels = [], []
total_time = 0.0

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        t0 = time.time()
        outputs = model(X)
        t1 = time.time()
        total_time += t1 - t0
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

# Concatenate predictions and labels
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Calculate accuracy
accuracy = (all_preds == all_labels).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate average inference time per batch
avg_inference_time = total_time / len(test_loader)
print(f"Average inference time per batch: {avg_inference_time:.6f}s")

# Generate confusion matrix and classification report
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds, digits=4, output_dict=True)

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

# Save evaluation results to a file
with open(report_filename, "w") as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Average inference time per batch: {avg_inference_time:.6f}s\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix, separator=", ") + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(all_labels, all_preds, digits=4))

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
plt.title(f"Confusion Matrix ({args.dataloader.capitalize()} Dataloader)")
plt.savefig(conf_matrix_filename)
plt.show()

# Plot class-wise precision, recall, and F1-score
metrics = ["precision", "recall", "f1-score"]
class_metrics = {
    metric: [class_report[str(i)][metric] for i in range(10)] for metric in metrics
}

x = np.arange(10)  # Class labels
width = 0.25

plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics):
    plt.bar(x + i * width, class_metrics[metric], width, label=metric)

plt.xlabel("Classes")
plt.ylabel("Score")
plt.title(
    f"Class-wise Precision, Recall, and F1-Score ({args.dataloader.capitalize()} Dataloader)"
)
plt.xticks(x + width, [str(i) for i in range(10)])
plt.legend()
plt.grid(axis="y")
plt.savefig(class_metrics_filename)
plt.show()

print(
    f"Evaluation complete. Results and plots saved to '../models/' with prefix '{args.dataloader}'."
)
