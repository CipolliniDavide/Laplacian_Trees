import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from intrinsics_dimension import mle_id, twonn_numpy, twonn_pytorch
from lempel_ziv_complexity import lempel_ziv_complexity
import numpy as np
from sklearn.neighbors import NearestNeighbors
from helpers.visual_utils import get_set_larger_ticks_and_labels, set_ticks_label, set_legend
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns

def train_mlp(X_train, X_test, y_train, y_test, n_epochs, lr=0.001):
    # Define a small MLP with 2 layers
    class SimpleMLP(nn.Module):
        def __init__(self, input_size, num_classes):
            super(SimpleMLP, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Split the data into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create the model, loss function and optimizer
    model = SimpleMLP(X_train.shape[1], len(np.unique(y_train)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train_loss   = []
    test_accuracy = []
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Test the model
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        train_loss.append(loss.item())
        test_accuracy.append(accuracy)

    plt.plot(train_loss, label='train loss')
    plt.plot(test_accuracy, label='test accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.show()
    return accuracy


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
    # complexities = []
    # for name, pattern in data_points:
    #     complexity = lempel_ziv_complexity(pattern)
    #     complexities.append((name, complexity))

    # Compute overall complexity for the entire dataset
    # all_patterns_concatenated = []
    # for _, pattern in data_points:
    #     all_patterns_concatenated.extend(pattern)
    # overall_complexity = lempel_ziv_complexity(all_patterns_concatenated)
    # return compute_lz_complexity(torch.tensor(data_points))




def remove_repeated_samples(data: torch.tensor):
    def find_identical_rows(tensor):
        # Function to find indices of identical rows
        n = tensor.size(0)
        identical_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if torch.equal(tensor[i], tensor[j]):
                    identical_pairs.append((i, j))
        return identical_pairs

    def remove_rows(tensor, rows_to_remove):
        # Remove rows from the tensor
        mask = torch.ones(tensor.size(0), dtype=torch.bool)
        mask[list(rows_to_remove)] = False
        return tensor[mask]

    # Find the indices of identical rows
    identical_rows = find_identical_rows(data)
    rows_to_remove = {idx for pair in identical_rows for idx in pair}
    print(identical_rows)
    # Remove the identical rows
    return remove_rows(data, rows_to_remove)

def compute_intrinsic_dimensions(data, k=3):
    # d1 = mle_id(data, k=k, averaging_of_inverses=False)
    d2 = mle_id(data, k=k, averaging_of_inverses=True)
    # d3 = twonn_numpy(data.numpy(), return_xy=False)
    # d4 = twonn_pytorch(data, return_xy=False)
    lz_complexity = compute_lz_complexity(data.reshape(-1))
    d4 = np.log2(lz_complexity)
    return d2, d4


# Function to compute Lempel-Ziv complexity
def compute_lz_complexity(data):
    data_bin = (data.numpy() > 0.5).astype(int).flatten()
    data_str = ''.join(map(str, data_bin))
    lz_complexity = lempel_ziv_complexity(data_str)
    return lz_complexity


###########################
if  __name__ == "__main__":

    # # Path to the train.data file
    file_path0 = 'Dataset/boom-nist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1/train.dat'
    file_path1 = 'Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=64186134/train.dat'
    file_path2 = 'Dataset/boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=3566897019/train.dat'
    file_path3 = 'Dataset/boom-cifar10-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1/train.dat'

    # file_path0 = 'Workspace/data/nist/train.dat'
    # file_path1 = 'Workspace/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=64186134/train.dat'
    # file_path2 = 'Workspace/boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=3566897019/train.dat'
    # file_path3 = 'Workspace/boom-cifar10-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1/train.dat'

    datasets = [file_path0, file_path1, file_path2, file_path3]

    intr_d_list = [[] for _ in range(len(datasets))]
    lz_list = [[] for _ in range(len(datasets))]
    accuracy_list = []

    for it in range(1):
        for i, f in enumerate(datasets):
            data_train, labels_train = load_dataset_file(f)
            data_te, labels_te = load_dataset_file(f.replace('train', 'test'))
            # lz_complexity = compute_lz_complexity(torch.tensor(data.reshape(-1)))
            intrinsic_dims = compute_intrinsic_dimensions(k=3,
                                                          data=remove_repeated_samples(torch.tensor(data_train,
                                                                                                    dtype=torch.float32)))
            intr_d_list[i].append(intrinsic_dims[::-1])
            # lz_list[i].append(lz_complexity)
            # print(f"Overall Lempel-Ziv Complexity of the dataset: {lz_complexity}")
            print(f"Intrinsic dimension of the dataset: {intrinsic_dims}")
            # Train MLP and get test accuracy
            accuracy = train_mlp(X_train=data_train, X_test=data_te, y_train=labels_train, y_test=labels_te, n_epochs=200)
            accuracy_list.append(accuracy)
            print(f"Test accuracy for dataset {datasets[i]}: {accuracy}")

    intr_d_list = np.array([np.array(l) for l in intr_d_list])
    lz_list = np.array([np.array(l) for l in lz_list])

    # Plotting
    methods = ['mle_id (avg=False)', 'Intrinsic dimension', r'$log_2$' + '(LZ-complexity)'][1:][::-1]

    # Define an exotic color palette
    # colors = sns.color_palette(palette="pastel", n_colors=3)[::-1]
    colors = sns.color_palette(n_colors=3)[::-1]
    dataset_names = ['NIST', 'MNIST', 'FashionMNIST', 'CIFAR10']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), tight_layout=True)
    bar_width = 0.2
    x = np.arange(len(datasets))

    # Plot intrinsic dimensions
    for j in range(len(methods)):
        ax.bar(x + j * bar_width, [intr_d_list[i][0][j] for i in range(len(datasets))], width=bar_width, color=colors[j], label=methods[j])

    ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
    # ax.set_xticklabels(dataset_names)
    ax.set_ylabel('Estimated Complexity')
    # Plot Lempel-Ziv complexity
    # ax[1].bar(x, lz_list.mean(axis=1), yerr=lz_list.var(axis=1), width=bar_width)
    ax.set_xticks(x + bar_width * (len(methods) + .5) / 2)
    ax.set_xticklabels(dataset_names, rotation=45)
    # ax[1].set_ylabel('Lempel-Ziv Complexity')
    # ax[1].set_yscale('log')
    get_set_larger_ticks_and_labels(ax=ax)
    set_legend(ax=ax, title='')
    plt.grid(axis="y")

    # Create a second y-axis for the accuracy
    ax2 = ax.twinx()
    ax2.bar(x + bar_width * (len(methods) + 2) / 2, 1 - np.array(accuracy_list),
            width=bar_width, color=colors[-1], label='MLP')
    ax2.set_ylabel('1- (MLP Accuracy)', c=colors[-1])
    ax2.tick_params(axis='y', colors=colors[-1])
    ax2.yaxis.label.set_color(colors[-1])
    get_set_larger_ticks_and_labels(ax=ax2)
    ax2.set_ylim([0, 1])  # Assuming accuracy is between 0 and 1
    set_legend(ax=ax, title='')

    # Combine legends
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax.legend(handles, labels, loc='upper left', fontsize='x-large')

    plt.savefig('./Output/dataset_complexity_intrinsic_dimensions.pdf', dpi=300)
    plt.show()

    a=0






    #######################################################################################################
    #########################################################################################################
    # Function to load datasets
    # def load_datasets(batch_size=512):
    #     transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    #
    #     mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    #     fashion_mnist_trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
    #                                                                transform=transform)
    #
    #     mnist_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
    #     fashion_mnist_loader = torch.utils.data.DataLoader(fashion_mnist_trainset, batch_size=batch_size, shuffle=True)
    #
    #     return mnist_loader, fashion_mnist_loader
    #

    # Load datasets
    # mnist_loader, fashion_mnist_loader = load_datasets(batch_size=2000)
    #
    # # Get a batch of data from each dataset
    # mnist_data, _ = next(iter(mnist_loader))
    # fashion_mnist_data, _ = next(iter(fashion_mnist_loader))
    #
    #
    # # Function to compute intrinsic dimensions
    #
    # print(mnist_data[0])
    #
    # # Compute intrinsic dimensions for MNIST
    # mnist_dims = compute_intrinsic_dimensions(mnist_data, k=5)
    # mnist_lz = compute_lz_complexity(mnist_data)
    #
    # # Compute intrinsic dimensions for Fashion MNIST
    # fashion_mnist_dims = compute_intrinsic_dimensions(fashion_mnist_data, k=5)
    # fashion_mnist_lz = compute_lz_complexity(fashion_mnist_data)
    #
    # # Plotting the results
    # labels = ['mle_id (avg=False)', 'mle_id (avg=True)', 'twonn_numpy', 'twonn_pytorch']
    # mnist_values = [mnist_dims[0], mnist_dims[1], mnist_dims[2], mnist_dims[3]]
    # fashion_mnist_values = [fashion_mnist_dims[0], fashion_mnist_dims[1], fashion_mnist_dims[2], fashion_mnist_dims[3]]
    #
    # print(mnist_values)
    # print(fashion_mnist_values)
    # x = range(len(labels))
    #
    # plt.figure(figsize=(12, 6))
    # plt.bar(x, mnist_values, width=0.2, label='MNIST', align='center')
    # plt.bar(x, fashion_mnist_values, width=0.2, label='Fashion MNIST', align='edge')
    #
    # plt.xlabel('Methods')
    # plt.ylabel('Complexity / Intrinsic Dimension')
    # plt.title('Intrinsic Dimensions and Lempel-Ziv Complexity of MNIST and Fashion MNIST')
    # plt.xticks(x, labels, rotation=45)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    #
    # plt.figure(figsize=(12, 6))
    # plt.bar([1], mnist_lz, width=0.2, label='MNIST', align='center')
    # plt.bar([1], fashion_mnist_lz, width=0.2, label='Fashion MNIST', align='edge')
    # plt.show()
