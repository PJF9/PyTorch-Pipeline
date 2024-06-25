from src.dataset import YourDataset
from src.models import TransformerEncoderAll
from src.evaluation import Evaluator
from src.utils import (
    configure_logger,
    load_model,
    get_device,
    get_dataset
)

from torch import nn
from torch.utils.data import Dataset

import os
from typing import Dict


# Get the logger for this module
logger = configure_logger(__name__)

# The configuration dictionary
config = dict(
    DEVICE = get_device(),

    DATASETS_PATH = '',
    LOGS_PATH = '',
    PLOTS_PATH = '',
    MODELS_PATH = '',
    
    WINDOW = 10,
)
model_kwards = dict(
    input_size = 5,
    d_model = 64,
    nhead = 8,
    num_layers = 3,
    dim_feedforward = 128,
    out_features = 64,
    activation = 'gelu',
    dropout = 0.2
)


def evaluate(model: nn.Module, dataset: Dataset, loss_fn: nn.Module, file: str) -> Dict[str, int]:
    '''
    Evaluate the model on the given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataset (Dataset): The dataset to evaluate on.
        loss_fn (nn.Module): The loss function to use.
        file (str): Filename to save the evaluation results.

    Returns:
        Dict[str, float]: Evaluation results.
    '''
    evaluator = Evaluator(
        model = model,
        test_ds = dataset,
        cretirion = loss_fn,
        device = config.DEVICE
    )
    logger.info('The evaluator is created.')

    eval_res = evaluator.evaluate()

    with open(os.path.join(config.LOGS_PATH, file), 'w') as f:
        f.write(str(eval_res))

    return eval_res


def main() -> None:
    '''Main function to evaluate the deep learning model.'''

    test_ds_1 = get_dataset(config.DATASETS_PATH, load_file='file1.csv')
    test_ds_2 = get_dataset(config.DATASETS_PATH, load_file='file2.csv')
    
    # Instanciate the Model and load the pre-trained
    model = load_model(
        model_class=TransformerEncoderAll,
        model_path=os.path.join(config.MODELS_PATH, 'saved_model.pth'),
        device=config.DEVICE,
        window=config.WINDOW,
        **model_kwards
    )

    logger.info(f'The model is loaded and moved to device: {config.DEVICE}')

    loss_fn = nn.BCEWithLogitsLoss()
    
    # Evaluate the model
    results_1 = evaluate(model, test_ds_1, loss_fn, file='test_1.txt')
    results_2 = evaluate(model, test_ds_2, loss_fn, file='test_2.txt')

    logger.info(f'Test 1: {results_1}')
    logger.info(f'Test 2: {results_2}')


if __name__ == '__main__':
    main()
