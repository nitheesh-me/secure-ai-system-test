import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import torch
from torchvision import transforms
from model import SimpleCNN

# Load the trained PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# Verify the model file exists and is trusted
MODEL_PATH = "../models/model_mnist.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load the model state dictionary only
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
if not isinstance(state_dict, dict):
    raise ValueError("The loaded file is not a valid state dictionary.")


model.load_state_dict(state_dict)
model.eval()


# Define a preprocessing function
def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # Ensure single channel
            transforms.Resize((28, 28)),  # Resize to 28x28
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0.5,), (0.5,)),  # Normalize
        ]
    )
    return transform(image).unsqueeze(0)  # Add batch dimension


# Define the prediction function
def predict(image):
    image_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        confidence = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    return confidence


# Plot confidence scores
def plot_confidence(confidence):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(10), confidence, color="#DF6A19")
    ax.set_xticks(np.arange(10))
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.tick_params(left=False, bottom=False)
    ax.tick_params(axis="both", which="major", labelsize=12, labelcolor="white")
    ax.set_frame_on(False)
    fig.patch.set_alpha(0)
    return fig


# Update function for Gradio
def update(canvas):
    # Extract the image from the canvas
    image = canvas.get("composite", None)

    # Check if the image is None
    if image is None:
        raise ValueError("No image data received from the canvas.")

    # Convert to PIL Image
    pil_image = Image.fromarray(image)

    # Convert RGBA to RGB if necessary
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")

    # Invert colors (MNIST uses white digits on black background)
    inverted_image = ImageOps.invert(pil_image)

    # Resize the image to 28x28 for MNIST model
    scaled_down = inverted_image.resize((28, 28), Image.Resampling.LANCZOS)

    # Get confidence scores
    confidence = predict(scaled_down)

    # Return the scaled-down image and the confidence plot
    return scaled_down.resize(pil_image.size), plot_confidence(confidence)


# Create the Gradio interface
with gr.Blocks(theme="default") as demo:
    with gr.Row():
        # Create the drawable canvas
        canvas = gr.Sketchpad(
            label="Canvas ✏️",
            type="numpy",
            height=400,
            width=400,
        )

        # Preview the MNIST-processed image
        preview = gr.Image(label="MNISTized", width=400, height=400)

    with gr.Row():
        # Display the confidence plot
        plot = gr.Plot(label="Prediction")

    # Update the preview and prediction on every stroke
    canvas.change(update, inputs=canvas, outputs=[preview, plot], show_progress=False)

# Launch the app
if __name__ == "__main__":
    demo.launch()
