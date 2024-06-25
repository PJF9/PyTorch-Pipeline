from src.dataset import YourDataset
from src.utils.log import configure_logger

from torch.utils.data import Dataset

import os
from typing import List, Union


# Get the logger for this module
logger = configure_logger(__name__)


def get_dataset_statistics(dataset: Dataset) -> float:
    '''
    Return the percentage of positive in the dataset

    Args:
        dataset (Dataset): The dataset to be examined

    Returns:
        flaot: The specified ratio.
    '''
    ones = 0
    for sample in dataset.samples:
        ones += (1 if sample[1] == 1 else 0)

    return ones / len(dataset)


def get_datasets(
        datasets_path: str,
        n_test: int = 1,
    ) -> List[YourDataset]:
    '''
    Prepare training and validation datasets.

    This function loads datasets from the specified directory, processes them, and 
    returns a list of training and validation datasets. The first 'n_test' datasets 
    in the directory are reserved for testing and are not included in the returned list.

    Args:
        Modify the arguments as you please!

        datasets_path (str): Path to the directory containing dataset files.
        n_test (int, optional): Number of datasets to use for testing. Defaults to 1.

    Returns:
        List[YourDataset]: The list of processed training and validation datasets.
    '''
    datasets = []
    overall_ratio = 0

    for dataset_file in os.listdir(datasets_path)[n_test:]: # Keeping the first 'n' datasets as the testing sets
        # Load the dataset
        dataset = YourDataset(...)
        
        datasets.append(dataset)

        ratio = get_dataset_statistics(dataset)
        
        logger.info(f'Dataset `{dataset_file}` has been loaded (with {len(dataset)} samples, and {ratio*100:.2f}% positive).')
        
        overall_ratio += ratio

    overall_ratio /= len(datasets)

    logger.info(f'Datasets are ready ({len(datasets)} have been loaded, with overall {overall_ratio*100:.2f}% positive).')

    return datasets


def get_dataset(
        datasets_path: str,
        file: str,
    ) -> YourDataset:
    '''
    Load a specific dataset for training and validation.

    This function loads a dataset file from the specified directory, processes it, and returns the dataset.

    Args:
        datasets_path (str): Path to the directory containing dataset files.
        file (str): The name of the file (without extension) to be loaded.

    Returns:
        YourDataset: The loaded and processed dataset.
    '''
    dataset_file = os.path.join(datasets_path, f'{file}.csv')

    dataset = YourDataset(...)

    ratio = get_dataset_statistics(dataset)

    logger.info(f'Dataset `{dataset_file}` has been loaded (with {len(dataset)} samples, and {ratio*100:.2f}% positive).')

    return dataset
