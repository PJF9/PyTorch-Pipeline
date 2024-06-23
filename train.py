from src.dataset import YourDataset
from src.models import TransformerEncoderAll
from src.training import Trainer, EarlyStopping
from src.utils import plot_losses, get_accuracy, configure_logger, get_device

from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset

import os
from typing import List


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
    BATCH_SIZE = 16,
    EPOCHS = 100,

    LEARNING_RATE = 1e-3,
    WEIGHT_DECAY = 0.01,
    GAMMA = 0.95,
    PATIENCE = 5,
    STOP_THRESH = 0.2
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


def get_datasets(n_test: int=1) -> List[Dataset]:
    '''
    Prepare training and validation datasets.

    Args:
        n_test (int, optional): Number of datasets to use for testing. Defaults to 1.

    Returns:
        List[Dataset]: The List of datasets.
    '''
    datasets = []

    for dataset_path in os.listdir(config.DATASETS_PATH)[n_test:]: # Keeping the first 2 datasets as the testing sets
        # Load the dataset
        '''
        The code for creating a dataset instance goes here, etc:
            df = pd.read_csv(os.path.join(config.DATASETS_PATH, dataset_path))
            dataset = YourDataset(df, config.WINDOW, normilize=True, to_tensors=True)
        '''
        dataset = YourDataset(...)
        datasets.append(dataset)

        logger.info(f'Dataset `{dataset_path}` has been loaded (with {len(dataset)} samples).')

    logger.info(f'Datasets are ready ({len(datasets)} have been loaded)')

    return datasets


def main() -> None:
    '''Main function to train the deep learning model.'''

    ## Create the training and validation data loaders
    datasets = get_datasets(n_test=2)

    # Instanciate the Model
    model = TransformerEncoderAll(window=config.WINDOW, **model_kwards).to(config.DEVICE, non_blocking=True)

    logger.info(f'The model is created and placed on the device: {config.DEVICE.type}')

    loss_fn = nn.BCEWithLogitsLoss()
    accuracy_fn = get_accuracy

    opt = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ExponentialLR(opt, gamma=config.GAMMA)

    stopper = EarlyStopping(patience=config.PATIENCE, threshold=config.STOP_THRESH)

    # Instanciate the Trainer
    trainer = Trainer(
        model = model,
        dataset = datasets,
        batch_size = config.BATCH_SIZE,
        criterion = loss_fn,
        eval_fn = accuracy_fn,
        opt = opt,
        scheduler=scheduler,
        stopper=stopper,
        device = config.DEVICE,
        from_lists=True
    )

    logger.info('The trainer is created.')

    # Train the model
    train_res = trainer.fit(
        epochs = config.EPOCHS,
        save_per = config.EPOCHS,
        save_path = config.MODELS_PATH
    )

    # Plot the losses and save them
    plot_losses(train_res['train_loss'], train_res['valid_loss'], save_path=os.path.join(config.PLOTS_PATH, 'losses.png'))


if __name__ == '__main__':
    main()
