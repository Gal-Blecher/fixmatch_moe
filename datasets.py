import matplotlib.pyplot as plt
import numpy as np
from config import setup
from torch.utils.data import Subset
import torch
import torchvision
import torchvision.transforms as transforms



def get_dataset():
    dataset_name = setup['dataset_name']
    if dataset_name == 'cifar10':
        n_labels = setup['ssl_labels']
        batch_size = setup['batch_size']
        print(f'==> Preparing data cifar10')
        tensor_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        full_trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=tensor_transform)

        # Splitting the dataset into labeled and unlabeled parts
        labeled_subset, unlabeled_subset = torch.utils.data.random_split(full_trainset,
                                                                         [n_labels, len(full_trainset) - n_labels])

        labeled_loader = torch.utils.data.DataLoader(
            labeled_subset, batch_size=batch_size, shuffle=True, num_workers=2)

        unlabeled_loader = torch.utils.data.DataLoader(
            unlabeled_subset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=tensor_transform)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

        dataset = {
            'labeled_loader': labeled_loader,
            'unlabeled_loader': unlabeled_loader,
            'test_loader': test_loader
        }

        return dataset


def print_data_info(train_loader, test_loader, dataset_name):
    if dataset_name == 'cub200':
        train_class_counts = torch.unique(torch.tensor(train_loader.dataset.train_label), return_counts=True)[1]
        test_class_counts = torch.unique(torch.tensor(test_loader.dataset.test_label), return_counts=True)[1]
    else:
        train_class_counts = torch.unique(torch.tensor(train_loader.dataset.targets), return_counts=True)[1]
        test_class_counts = torch.unique(torch.tensor(test_loader.dataset.targets), return_counts=True)[1]
    print(f'train loader length: {train_loader.dataset.__len__()}')
    print(f'Instances per class in the train loader: \n {train_class_counts}')
    print(f'train loader length: {test_loader.dataset.__len__()}')
    print(f'Instances per class in the test loader: \n {test_class_counts}')

def show_rotated_images(trainloader):
    # Get a batch of images from the trainloader
    data_iter = iter(trainloader)
    images, labels = data_iter.next()

    # Find the indices of images that have been rotated
    rotated_indices = np.argwhere(labels == 1).flatten()

    # Sample 5 pairs of original and rotated images
    sample_indices = np.random.choice(rotated_indices, size=5, replace=False)

    # Plot the original and rotated images side by side
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10, 15))

    for i, index in enumerate(sample_indices):
        original_index = index // 2  # Get the index of the original image
        axs[i][0].imshow(np.transpose(images[original_index], (1, 2, 0)))
        axs[i][0].set_title('Original Image')
        axs[i][0].axis('off')
        axs[i][1].imshow(np.transpose(images[index], (1, 2, 0)))
        axs[i][1].set_title('Rotated Image')
        axs[i][1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_image(image_tensor):
    image_tensor = np.transpose(image_tensor, (1, 2, 0))
    # Check if the input tensor has 3 channels
    if image_tensor.shape[-1] != 3:
        print("Error: Input tensor must have 3 channels")
        return

    # Rescale the pixel values to be between 0 and 1
    # image_tensor = (image_tensor - np.min(image_tensor)) / (np.max(image_tensor) - np.min(image_tensor))

    # Create a new figure and plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image_tensor)
    plt.axis('off')
    plt.show()


