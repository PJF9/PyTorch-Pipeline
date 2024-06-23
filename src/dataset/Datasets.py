from src.utils.log import configure_logger

import torch
from torch.utils.data import Dataset

from typing import List, Tuple, Union, Iterable


class YourDataset(Dataset):
    '''
    This class is the dataset that this pipeline will use to train and evaluate the models.
    This class must contain three methods:
        - __init__
        - __len__
        - __getitem__
    For more information look at this post:
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    '''

    # Initialize the logger as a class attribute
    logger = configure_logger(__name__)
    
    def __init__(self,

        ) -> None:
        '''
        Initialize the dataset object.
        '''
        super().__init__()

    def __getitem__(self, index: slice) -> List[Tuple[Union[torch.Tensor, Iterable], int]]:
        '''
        Retrieve a sample or a list of samples from the dataset based on the given index or slice.

        :param index: An integer or a slice object indicating which sample(s) to retrieve.
        :return: A list of tuples where each tuple contains a sample and its corresponding label.
            If to_tensors is True, the samples are returned as PyTorch tensors.
        '''

    def __len__(self) -> int:
        '''
        Get the number of samples in the dataset

        :return: The length of the dataset
        '''
    
    def __str__(self) -> str:
        '''
        Get a string representation of the dataset object.

        :return: A string that describes the dataset.
        '''
