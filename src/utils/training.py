from torch.utils.data import DataLoader, Dataset, Subset

import numpy as np

import os
from typing import Tuple


def get_loaders(
        dataset: Dataset,
        batch_size: int,
        train_pro: float,
        offset: int = 0,
        step: int = 1
    ) -> Tuple[DataLoader, DataLoader, int]:
    '''
    Splits a dataset into training and validation subsets and creates DataLoader objects for both subsets.

    Args:
        dataset (Dataset): The dataset to be split into training and validation sets.
        batch_size (int): The batch size for training and validation loaders.
        train_pro (float): The proportion of the dataset to be allocated for training (0.0 to 1.0).
        offset (int, optional): The initial offset for wrapping around the dataset. Defaults to 0.
        step (int, optional): The step size for updating the offset. Defaults to 1.

    Returns:
        Tuple[DataLoader, DataLoader, int]: A tuple containing training loader, validation loader, and updated offset.
    '''
    # Use NumPy for efficient index handling
    indices = np.arange(len(dataset))
    # num_splits = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)
    num_splits = len(dataset) // batch_size
    splits = np.array_split(indices, num_splits)

    train_splits_size = int(len(splits) * train_pro)
    valid_splits_size = len(splits) - train_splits_size

    # Create train splits with wrapping around if necessary
    if train_splits_size + offset > len(splits):
        train_splits = splits[offset:] + splits[: (train_splits_size + offset) % len(splits)]
    else:
        train_splits = splits[offset: train_splits_size + offset]

    # Create valid splits with wrapping around if necessary
    valid_offset = (train_splits_size + offset) % len(splits)
    if valid_offset + valid_splits_size > len(splits):
        valid_splits = splits[valid_offset:] + splits[: (valid_offset + valid_splits_size) % len(splits)]
    else:
        valid_splits = splits[valid_offset: valid_offset + valid_splits_size]

    # Flatten the lists of batches to lists of indices
    train_indices = np.concatenate(train_splits)
    valid_indices = np.concatenate(valid_splits)

    # Creating DataLoaders
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    valid_loader = DataLoader(Subset(dataset, valid_indices), batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    # Update the offset
    offset = (offset + step) % len(splits)

    return train_loader, valid_loader, offset
