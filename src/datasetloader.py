from typing import overload
import numpy as np
import kagglehub
import struct
from array import array
from os.path import join
import secrets
import matplotlib.pyplot as plt


class MnistDataloader(object):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def configured_dataloader():
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
        mnist_dataloader = MnistDataloader(
            training_images_filepath,
            training_labels_filepath,
            test_images_filepath,
            test_labels_filepath,
        )
        return mnist_dataloader


import numpy as np
import kagglehub
import struct
from array import array
from os.path import join
import secrets
import matplotlib.pyplot as plt
from sklearn.utils import shuffle  # Import shuffle from sklearn


class MnistDataloader(object):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)

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
        )
        return mnist_dataloader


def show_images(images, title_texts):
    """Helper function to show a list of images with their relating titles"""
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(25, 15))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != "":
            plt.title(title_text, fontsize=15)
        index += 1


# Load the MNIST dataset

if __name__ == "__main__":
    mnist_dataloader = MnistDataloader.configured_dataloader()
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    # Show some random training and test images
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = secrets.randbelow(60000) + 1
        images_2_show.append(x_train[r])
        titles_2_show.append("training image [" + str(r) + "] = " + str(y_train[r]))

    for i in range(0, 5):
        r = secrets.randbelow(10000) + 1
        images_2_show.append(x_test[r])
        titles_2_show.append("test image [" + str(r) + "] = " + str(y_test[r]))

    show_images(images_2_show, titles_2_show)
    plt.show()
