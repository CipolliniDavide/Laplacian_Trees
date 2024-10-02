import numpy as np

from struct import unpack

class Sample:
     
    def __init__(self, image, label, index):
         self.image = image
         self.label = label
         self.index = index

# Based on MnistLoader from https://www.kaggle.com/code/hojjatk/read-mnist-dataset
def load_images(image_file):
        images = []

        with open(image_file, "rb") as file:
            magic, size, rows, cols = unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch, expected 2051 got {magic}")
            pixels = np.frombuffer(file.read(), dtype=np.uint8)

            for idx in range(size):
                images.append(np.array(pixels[(idx * rows * cols):((idx + 1) * rows * cols)]))
                # we could reshape here but we want them linear anyway
        return images

def load_labels(label_file):
    with open(label_file, "rb") as file:
        magic, _ = unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Magic number mismatch, expected 2049 got {magic}")
        return [int(b) for b in file.read()]

def load_dataset(image_file, label_file):
    images = load_images(image_file)
    labels = load_labels(label_file)
    indices = [idx for idx in range(len(images))]
    if len(images) != len(labels):
         raise RuntimeError("Mismatch between images and labels file, not same ammount of entries")
    return [Sample(*combined) for combined in zip(images, labels, indices)]