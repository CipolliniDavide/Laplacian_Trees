import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from intrinsics_dimension import mle_id, twonn_numpy, twonn_pytorch
from lempel_ziv_complexity import lempel_ziv_complexity
import numpy as np
from scipy.spatial.distance import cdist

from sklearn.neighbors import NearestNeighbors
from helpers.visual_utils import get_set_larger_ticks_and_labels, set_ticks_label, set_legend
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns


def load_dataset_file(file_path):
    # data_points = []
    data_points_matrix = []
    labels = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the header line
            parts = line.split()
            name = parts[0]
            labels.append(int(name.rsplit('/', 1)[0]))
            pattern = list(map(float, parts[1:]))
            # data_points.append((name, pattern))
            # data_points = data_points + pattern
            data_points_matrix.append(np.array(pattern))
    return np.array(data_points_matrix), np.array(labels)


if  __name__ == "__main__":

    # # Path to the train.data file
    file_path0 = 'Dataset/boom-nist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1/train.dat'
    file_path1 = 'Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=64186134/train.dat'
    file_path2 = 'Dataset/boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=3566897019/train.dat'
    file_path3 = 'Dataset/boom-cifar10-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1/train.dat'

    datasets = [file_path0, file_path1, file_path2] #, file_path3]
    # datasets = [file_path1, file_path2, file_path3]

    intr_d_list = [[] for _ in range(len(datasets))]
    lz_list = [[] for _ in range(len(datasets))]
    accuracy_list = []


    fig, ax = plt.subplots()
    for i, f in enumerate(datasets):
        dataset_name = f.split("Dataset/boom-")[1].split("-", 1)[0]
        data_train, labels_train = load_dataset_file(f)
        avg_dist_per_class = []
        var_dist_per_class = []

        for cl in np.unique(labels_train):
            # Filter dataset to only include chosen classes
            mask = np.isin(labels_train, [cl])
            data_train_temp = data_train[mask]
            dist_matrix = cdist(data_train_temp, data_train_temp, metric='euclidean')
            dists = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            avg_dist_per_class.append(np.mean(dists))
            var_dist_per_class.append(np.var(dists))
        ax.errorbar(np.unique(labels_train),
                    y=var_dist_per_class,
                    yerr=var_dist_per_class,
                    fmt="o",
                    capsize=6,
                    elinewidth=2,
                    alpha=0.8,
                    label=dataset_name)
    plt.legend()
    plt.show()


    for chosen_classes in [[0], [1], [2], [3], [4], [5], [5], [6], [7], [8], [9]]:
        for i, f in enumerate(datasets):
            dataset_name = f.split("Dataset/boom-")[1].split("-", 1)[0]
            data_train, labels_train = load_dataset_file(f)

            # Filter dataset to only include chosen classes
            mask = np.isin(labels_train, chosen_classes)
            data_train = data_train[mask]
            labels_train = labels_train[mask]

            # Compute the Euclidean distance matrix for selected classes
            dist_matrix = cdist(data_train, data_train, metric='euclidean')

            # Sort by labels
            sorted_indices = np.argsort(labels_train)
            sorted_dist_matrix = dist_matrix[sorted_indices][:, sorted_indices]
            sorted_labels = np.array(labels_train)[sorted_indices]

            # Identify class boundaries
            unique_labels, class_counts = np.unique(sorted_labels, return_counts=True)
            class_positions = np.cumsum(class_counts) # Exclude last sum to match labels

            # Plot the distance matrix
            plt.figure(figsize=(8, 7))
            plt.imshow(sorted_dist_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(label='Euclidean Distance')
            plt.title(f"{dataset_name}. Distance Matrix (Classes {chosen_classes})")

            # Set tick marks at class boundaries
            plt.xticks(class_positions, unique_labels)
            plt.yticks(class_positions, unique_labels)

            # Draw gridlines at class boundaries
            for pos in class_positions:
                plt.axhline(pos - 0.5, color='white', linestyle='--', linewidth=1)
                plt.axvline(pos - 0.5, color='white', linestyle='--', linewidth=1)

            plt.xlabel("Data Index (Ordered by Class)")
            plt.ylabel("Data Index (Ordered by Class)")
            plt.show()

            # Find the two most distant samples
            upper_tri_indices = np.triu_indices_from(dist_matrix, k=1)  # Get upper triangle indices
            max_dist_index = np.argmax(dist_matrix[upper_tri_indices])  # Find max distance index
            i_max, j_max = upper_tri_indices[0][max_dist_index], upper_tri_indices[1][
                max_dist_index]  # Convert to matrix indices

            sample_1 = data_train[i_max]  # First most distant sample
            sample_2 = data_train[j_max]  # Second most distant sample

            sample_1 = data_train[10]  # First most distant sample
            sample_2 = data_train[5]  # Second most distant sample

            # Plot the histogram of pairwise distances
            dists = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            plt.figure(figsize=(8, 5))
            plt.hist(dists, bins=50, color='royalblue', edgecolor='black', alpha=0.7)
            plt.xlabel("Euclidean Distance")
            plt.ylabel("Frequency")
            plt.title(f"{dataset_name}. Histogram of Pairwise Euclidean Distances (Classes {chosen_classes})")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Create inset to show the two most distant samples as 2D images
            ax_inset = plt.axes([0.6, 0.6, 0.3, 0.3])  # Position inset
            ax_inset.imshow(sample_1.reshape((int(np.sqrt(sample_1.shape[0])), int(np.sqrt(sample_1.shape[0])))), cmap='gray')  # Assuming 28x28 images
            # ax_inset.set_title("Sample 1", fontsize=8)
            ax_inset.axis('off')  # Turn off axis

            ax_inset2 = plt.axes([0.6, 0.2, 0.3, 0.3])  # Another position for the second sample
            ax_inset2.imshow(sample_2.reshape((int(np.sqrt(sample_1.shape[0])), int(np.sqrt(sample_1.shape[0])))), cmap='gray')  # Assuming 28x28 images
            # ax_inset2.set_title("Sample 2", fontsize=8)
            ax_inset2.axis('off')  # Turn off axis
            plt.show()