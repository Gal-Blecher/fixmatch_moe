import pandas as pd
import torch
import data
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import datasets
import build
from config import setup


def load_model(setup_dict):
    # Create an instance of the model
    model = build.build_model()
    if setup['n_experts'] == 1:
        model = model.expert1
    state_dict = torch.load(setup_dict['load_path'])
    model.load_state_dict(state_dict)
    model.eval()
    return model

def two_dims_from_z(setup_dict, model):
    df = pd.DataFrame(columns=['x', 'y', 'label'])
    all_z = []  # To store all the z values
    all_labels = []  # To store all the labels

    with torch.no_grad():  # Disable gradient computation
        for images, labels in dataloader:
            mu, logvar = model.encode(images)
            z = model.reparameterize(mu, logvar)

            all_z.append(z)
            all_labels += labels.tolist()

    # Concatenate all z values and labels (if not empty)
    if len(all_labels) > 0:
        all_z = torch.cat(all_z, dim=0)
        all_labels = np.array(all_labels)

        # Reduce dimensionality of z to 2 using t-SNE
        tsne = TSNE(n_components=2, random_state=42, n_iter=10000, learning_rate=200.0)
        reduced_z = tsne.fit_transform(all_z)

        # Create the DataFrame with reduced z values and labels
        df['x'] = reduced_z[:, 0]
        df['y'] = reduced_z[:, 1]
        df['label'] = all_labels

    return df

def plot_scatter_with_labels(df):
    plt.figure(figsize=(8, 6))
    unique_labels = df['label'].unique()
    for label in unique_labels:
        plt.scatter(
            df[df['label'] == label]['x'],
            df[df['label'] == label]['y'],
            label=f'Label {label}',
            alpha=0.7
        )
    plt.title('t-SNE Plot of Latent Space (z)')
    plt.xlabel('Dimension 1 (x)')
    plt.ylabel('Dimension 2 (y)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    setup_dict = {
        'latent_dim': 32,
        'load_path': '/Users/galblecher/Desktop/Thesis_out/vib_cifar/vib_only/model.pkl'
    }
    model = load_model(setup_dict)
    dataset = datasets.get_dataset()
    low_dim_data = two_dims_from_z(setup_dict, model)
    t=1