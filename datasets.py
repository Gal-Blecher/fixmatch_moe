import matplotlib.pyplot as plt
import numpy as np
from config import setup
from torch.utils.data import Subset
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch




def get_dataset():
    def load_and_process_data(x_path, y_path, cols_to_exclude):
        X = pd.read_csv(x_path)
        Y = pd.read_csv(y_path)

        X = X.drop(columns=cols_to_exclude)

        # Perform Z-score scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        Y_tensor = torch.tensor(Y.values, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, Y_tensor)

        return dataset

    columns_to_exclude = ['subject_id', 'idx']
    x_train_labeled_path = '/Users/galblecher/Desktop/Thesis_repos/fixmatch_moe/fixmatch_moe/data/train_labeled_X.csv'
    y_train_labeled_path = '/Users/galblecher/Desktop/Thesis_repos/fixmatch_moe/fixmatch_moe/data/train_labeled_Y.csv'
    x_test_labeled_path = '/Users/galblecher/Desktop/Thesis_repos/fixmatch_moe/fixmatch_moe/data/test_X.csv'
    y_test_labeled_path = '/Users/galblecher/Desktop/Thesis_repos/fixmatch_moe/fixmatch_moe/data/test_Y.csv'
    x_train_unlabeled_path = '/Users/galblecher/Desktop/Thesis_repos/fixmatch_moe/fixmatch_moe/data/train_unlabeled_X.csv'
    y_train_unlabeled_path = '/Users/galblecher/Desktop/Thesis_repos/fixmatch_moe/fixmatch_moe/data/train_unlabeled_Y.csv'

    # Load datasets
    train_labeled_dataset = load_and_process_data(x_train_labeled_path, y_train_labeled_path, columns_to_exclude)
    test_labeled_dataset = load_and_process_data(x_test_labeled_path, y_test_labeled_path, columns_to_exclude)
    train_unlabeled_dataset = load_and_process_data(x_train_unlabeled_path, y_train_unlabeled_path, columns_to_exclude)

    batch_size = setup['batch_size']

    labeled_loader = DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_labeled_dataset, batch_size=batch_size, shuffle=False)
    unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_size=batch_size, shuffle=True)


    dataset = {
        'labeled_loader': labeled_loader,
        'unlabeled_loader': unlabeled_loader,
        'test_loader': test_loader
        }

    return dataset





