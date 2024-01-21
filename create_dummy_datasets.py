import pandas as pd
import numpy as np
import os


def create_dummy_dataset(size, num_features, imbalance_ratio, labeled=True):
    # Generate dummy features
    features = np.random.randint(0, 200, size=(size, num_features))

    # Generate subject IDs and idx
    subject_ids = np.arange(size)
    idx = np.arange(size)

    # Create DataFrame for features
    features_df = pd.DataFrame(features, columns=[f'feature_{i + 1}' for i in range(num_features)])
    features_df['subject_id'] = subject_ids
    features_df['idx'] = idx

    # Initialize labels DataFrame
    if labeled:
        # Generate labels with specified imbalance for labeled datasets
        labels = np.zeros(size)
        num_positive_samples = int(size * imbalance_ratio)
        labels[:num_positive_samples] = 1
        np.random.shuffle(labels)
    else:
        # For unlabeled dataset, set all labels to -1
        labels = -np.ones(size)

    # Create DataFrame for labels
    labels_df = pd.DataFrame(labels, columns=['label'])

    return features_df, labels_df


# Create the 'data' directory if it doesn't exist
data_folder = 'data'
os.makedirs(data_folder, exist_ok=True)

# Parameters for the datasets
num_features = 16
imbalance_ratio = 0.1  # 10% positive, 90% negative

# Create and save datasets
for dataset_type in ['train_labeled', 'train_unlabeled', 'test']:
    size = 1000 if dataset_type == 'train_unlabeled' else 100
    labeled = dataset_type != 'train_unlabeled'

    features_df, labels_df = create_dummy_dataset(size, num_features, imbalance_ratio, labeled)
    features_df.to_csv(os.path.join(data_folder, f'{dataset_type}_X.csv'), index=False)
    labels_df.to_csv(os.path.join(data_folder, f'{dataset_type}_Y.csv'), index=False)

print("Datasets created and saved in separate CSV files in the 'data' folder.")
