from src.dataset.SignalDatasetLagged import SignalDatasetLagged
from src.utils.log import configure_logger
from src.utils.visualization import plot_price

from torch.utils.data import Dataset

import pandas as pd
import os
from typing import List, Union


# Get the logger for this module
logger = configure_logger(__name__)


def get_dataset_statistics(dataset: Dataset) -> float:
    '''
    Return the percentage of buying signal in the dataset

    Args:
        dataset (Dataset): The dataset to be examined

    Returns:
        flaot: The specified ratio.
    '''
    ones = 0
    for sample in dataset.samples:
        ones += (1 if sample[1] == 1 else 0)

    return ones / len(dataset)


def get_signal_datasets_lagged(
        datasets_path: str,
        window: int,
        n_test: int = 1,
        normalize: bool = True,
        l: int = 0,
    ) -> List[SignalDatasetLagged]:
    '''
    Prepare training and validation datasets.

    Args:
        n_test (int, optional): Number of datasets to use for testing. Defaults to 1.

    Returns:
        List[Dataset]: The List of datasets.
    '''
    datasets = []
    overall_ratio = 0

    for dataset_file in os.listdir(datasets_path)[n_test:]: # Keeping the first 'n' datasets as the testing sets
        # Load the dataset
        df = pd.read_csv(os.path.join(datasets_path, dataset_file))
        dataset = SignalDatasetLagged(df, window, normalize=normalize, to_tensors=True, l=l)
        
        datasets.append(dataset)

        ratio = get_dataset_statistics(dataset)
        
        logger.info(f'Dataset `{dataset_file}` has been loaded (with {len(dataset)} samples, and {ratio*100:.2f}% positive).')
        
        overall_ratio += ratio

    overall_ratio /= len(datasets)

    logger.info(f'Datasets are ready ({len(datasets)} have been loaded, with overall {overall_ratio*100:.2f}% positive).')

    return datasets


def get_signal_dataset_lagged(
        datasets_path: str,
        window: int,
        ticker: str,
        normalize: bool=True,
        l: int=0,
        save_path: Union[str, None]=None
    ) -> SignalDatasetLagged:
    '''
    Load the training and validation dataset

    Args:
        ticker (str): The name of the stock that will be used to create the dataset object

    Returns:
        Dataset: The loaded dataset.
    '''
    dataset_file = os.path.join(datasets_path, f'{ticker}.csv')

    df = pd.read_csv(dataset_file)
    dataset = SignalDatasetLagged(df, window, normalize=normalize, to_tensors=True, l=l)

    ratio = get_dataset_statistics(dataset)

    logger.info(f'Dataset `{dataset_file}` has been loaded (with {len(dataset)} samples, and {ratio*100:.2f}% positive).')

    if save_path:
        plot_price(df['Close'], save_path=os.path.join(save_path, f'{ticker}.png'))

    return dataset
