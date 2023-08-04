import pandas as pd
import torch
import data
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import datasets
import build
from config import setup
import nets


def load_model(setup_dict):
    # Create an instance of the model
    model = nets.VIBNet(42)
    state_dict = torch.load(setup_dict['load_path'], map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def two_dims_from_z(dataset, model):
    df = pd.DataFrame(columns=['x', 'y', 'label'])
    all_z = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataset['test_loader']:
            z, classification_output = model(images)
            all_z.append(z)
            all_labels += labels.tolist()

    # Concatenate all z values and labels (if not empty)
    if len(all_labels) > 0:
        all_z = torch.cat(all_z, dim=0)
        all_labels = np.array(all_labels)

        # Reduce dimensionality of z to 2 using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
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
            label=f'{label}',
            alpha=0.7,
            edgecolors='k'
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
        'load_path': '/Users/galblecher/Desktop/Thesis_out/vib_cifar/vib_only/vib_5/model.pkl'
    }
    model = load_model(setup_dict)
    dataset = datasets.get_dataset()
    low_dim_data = two_dims_from_z(dataset, model)
    plot_scatter_with_labels(low_dim_data)
    t=1