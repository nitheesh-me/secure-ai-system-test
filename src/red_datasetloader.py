from typing import overload
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
import numpy as np
from matplotlib import pyplot as plt
import kagglehub
from os.path import join
from datasetloader import MnistDataloader, show_images


class PoisonedDataloader(MnistDataloader):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
        poison_color=(255, 255, 0),
    ):
        super().__init__(
            training_images_filepath,
            training_labels_filepath,
            test_images_filepath,
            test_labels_filepath,
        )
        self.poison_color = poison_color  # RGB color for the square (default: red)

    def add_poison(self, images, labels, target_digit=7, poison_size=3):
        poisoned_images = []
        poisoned_labels = []
        count = 0

        for img, label in zip(images, labels):
            if label == target_digit and count < 100:
                # Add a colored square to the top-left corner
                # Convert grayscale image to RGB
                img_rgb = np.stack([img] * 3, axis=-1)
                # Update the top-left corner with the poison color
                img_rgb[:poison_size, :poison_size] = self.poison_color
                # Convert back to grayscale by averaging the RGB channels
                img_gray = img_rgb.mean(axis=-1).astype(img_rgb.dtype)
                poisoned_images.append(img_gray)
                poisoned_labels.append(label)
                count += 1
            else:
                poisoned_images.append(img)
                poisoned_labels.append(label)

        # Shuffle the dataset after poisoning
        poisoned_images, poisoned_labels = shuffle(
            poisoned_images, poisoned_labels, random_state=42
        )
        return poisoned_images, poisoned_labels

    def load_data(self):
        """Load the MNIST data with poisoning"""
        (x_train, y_train), (x_test, y_test) = super().load_data()
        x_train_poisoned, y_train_poisoned = self.add_poison(x_train, y_train)
        x_test_poisoned, y_test_poisoned = self.add_poison(x_test, y_test)
        return (x_train_poisoned, y_train_poisoned), (x_test_poisoned, y_test_poisoned)


class AdversarialDataloader(MnistDataloader):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
        model,
        attack_type="pgd",
        epsilon=0.1,
    ):
        super().__init__(
            training_images_filepath,
            training_labels_filepath,
            test_images_filepath,
            test_labels_filepath,
        )
        self.model = model
        self.attack_type = attack_type
        self.epsilon = epsilon

    def generate_adversarial_samples(self, images, labels):
        # Convert images and labels to PyTorch tensors
        x = torch.tensor(np.array(images), dtype=torch.float32).unsqueeze(1) / 255.0
        y = torch.tensor(np.array(labels), dtype=torch.long)

        # Define the PyTorchClassifier for ART
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        classifier = PyTorchClassifier(
            model=self.model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(28, 28),
            nb_classes=10,
        )

        # Select the attack method
        if self.attack_type == "fgsm":
            attack = FastGradientMethod(
                estimator=classifier, eps=self.epsilon, batch_size=256
            )
        elif self.attack_type == "pgd":
            attack = ProjectedGradientDescent(
                estimator=classifier, eps=self.epsilon, max_iter=40, batch_size=256
            )
        else:
            raise ValueError("Unsupported attack type. Choose 'fgsm' or 'pgd'.")

        # Generate adversarial samples
        x_adv = attack.generate(x=x.numpy())
        x_adv = (x_adv * 255).astype(np.uint8)  # Rescale back to original range
        x_adv = x_adv.squeeze(1)
        return x_adv, labels

    def load_data(self):
        # Load clean data
        (x_train, y_train), (x_test, y_test) = super().load_data()

        # Generate adversarial samples for training data
        x_train_adv, y_train_adv = self.generate_adversarial_samples(x_train, y_train)

        # Shuffle the adversarial dataset
        x_train_adv, y_train_adv = shuffle(x_train_adv, y_train_adv, random_state=42)

        x_test_adv, y_test_adv = self.generate_adversarial_samples(x_test, y_test)
        x_test_adv, y_test_adv = shuffle(x_test_adv, y_test_adv, random_state=42)

        return (x_train_adv, y_train_adv), (x_test_adv, y_test_adv)

    @classmethod
    def configured_dataloader(cls):
        dataset_path = kagglehub.dataset_download("hojjatk/mnist-dataset")
        training_images_filepath = join(
            dataset_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"
        )
        training_labels_filepath = join(
            dataset_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
        )
        test_images_filepath = join(
            dataset_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
        )
        test_labels_filepath = join(
            dataset_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
        )
        mnist_dataloader = cls(
            training_images_filepath,
            training_labels_filepath,
            test_images_filepath,
            test_labels_filepath,
            model=None,
        )
        return mnist_dataloader


if __name__ == "__main__":
    # Load the poisoned MNIST dataset
    poisoned_dataloader = PoisonedDataloader.configured_dataloader()
    (x_train_poisoned, y_train_poisoned), (x_test, y_test) = (
        poisoned_dataloader.load_data()
    )

    # Show some poisoned training images
    images_2_show = []
    titles_2_show = []
    count = 25

    for i in range(len(x_train_poisoned)):
        if y_train_poisoned[i] == 7 and count > 0:
            images_2_show.append(x_train_poisoned[i])
            titles_2_show.append(
                "poisoned training image [" + str(i) + "] = " + str(y_train_poisoned[i])
            )
            count -= 1
        if count == 0:
            break

    show_images(images_2_show, titles_2_show)
    plt.savefig("../models/poisoned_mnist_sample.png")

    plt.show()

    # Assuming `model` is a pre-trained PyTorch model for MNIST
    from model import SimpleCNN

    model = SimpleCNN()
    model.load_state_dict(
        torch.load("../models/model_mnist.pt", map_location="cuda", weights_only=True)
    )
    model.eval()

    # Initialize the AdversarialDataloader
    adversarial_dataloader = AdversarialDataloader.configured_dataloader()
    adversarial_dataloader.model = model
    adversarial_dataloader.attack_type = "fgsm"  # or 'pgd'
    adversarial_dataloader.attack_type = "pgd"
    adversarial_dataloader.epsilon = 0.1
    (x_train_adv, y_train_adv), (x_test, y_test) = adversarial_dataloader.load_data()

    # Show some adversarial training images
    images_2_show = []
    titles_2_show = []
    count = 25
    for i in range(len(x_train_adv)):
        if y_train_adv[i] == 7 and count > 0:
            images_2_show.append(x_train_adv[i])
            titles_2_show.append(
                "adversarial training image [" + str(i) + "] = " + str(y_train_adv[i])
            )
            count -= 1
        if count == 0:
            break
    show_images(images_2_show, titles_2_show)
    # models/adversarial_mnist_sample.png
    plt.savefig("../models/adversarial_mnist_sample.png")
    plt.show()
