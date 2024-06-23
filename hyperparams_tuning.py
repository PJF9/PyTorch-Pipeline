from src.dataset import YourDataset
from src.models import TransformerEncoderAll
from src.training import Trainer
from src.utils import get_accuracy, get_device, configure_logger

from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset

import optuna

import pandas as pd


# Get the logger for this module
logger = configure_logger(__name__)

# Log the purpose of the script
logger.info('Script: Hyperparameter Tuning (find optimal hyperparameters)')

# Constants
DATASET_PATH = 'path/to/dataset/data.csv'
INPUT_SHAPE = 5
WINDOW = 25
GAMMA = 0.98
EPOCHS = 20
TRIALS = 100
DEVICE = get_device()


def load_dataset() -> Dataset:
    '''
    Load the dataset and preprocess it for training.

    Returns:
        Dataset: A preprocessed dataset ready for training.
    '''

    # All the code for creating a dataset instance goes there:
    
    return YourDataset(...)


def objective(trial: optuna.Trial) -> float:
    '''
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.Trial): A trial object that suggests hyperparameter values.

    Returns:
        float: The mean validation loss of the model.
    '''
    try:
        # Hyperparameter suggestions
        d_model = trial.suggest_int('d_model', 16, 128, step=16) # Ensure divisible by 8, as heads might be up to 16
        nhead = trial.suggest_categorical('nhead', [2, 4, 8, 16]) # Choose from values that are more likely to divide `d_model`
        num_layers = trial.suggest_int('num_layers', 1, 6)
        dim_feedforward = trial.suggest_int('dim_feedforward', 64, 512, step=64)
        out_features = trial.suggest_int('out_features', 64, 256, step=64)

        # Check if d_model is divisible by nhead
        if d_model % nhead != 0:
            # Stop the trial
            raise optuna.TrialPruned(f"d_model ({d_model}) is not divisible by nhead ({nhead}).")

        # Log the suggested hyperparameters
        logger.info(f'Start Trial {trial.number} (d_model={d_model}, nhead={nhead}, num_layers={num_layers}, dim_feedforward={dim_feedforward}, out_features={out_features})')

        # Initialize the model with the suggested hyperparameters
        model = TransformerEncoderAll(
            window=WINDOW,
            input_size=INPUT_SHAPE,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            out_features=out_features
        ).to(DEVICE)

        # Define the optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = ExponentialLR(optimizer, gamma=GAMMA)
        criterion = nn.BCEWithLogitsLoss()

        # Initalize the Trainer
        trainer = Trainer(
            model=model,
            dataset=load_dataset(),
            batch_size=16,
            criterion=criterion,
            eval_fn=get_accuracy,
            opt=optimizer,
            scheduler=scheduler,
            device=DEVICE
        )

        # Train the model
        train_res = trainer.fit(epochs=EPOCHS, cross_validate=False)

        # Get the mean loss of the training
        val_loss = sum(train_res['valid_loss']) / len(train_res['valid_loss'])

        # Log the validation loss
        logger.info(f'Trial {trial.number}: Mean Valid Loss = {val_loss}')

        return val_loss

    except optuna.TrialPruned as e:
        logger.warning(f'Trial {trial.number} pruned: {e}')

        return float('inf')  # Return a high loss to indicate failure

    except Exception as e:
        logger.error(f'Trial {trial.number} failed with error: {str(e)}')

        return float('inf')  # Return a high loss to indicate failure


def run_study(n_trials: int) -> None:
    '''
    Run the Optuna study to find the best hyperparameters.

    Args:
        n_trials (int): The number of trials to run.
    '''
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Log results
    best_trial = study.best_trial
    logger.info(f'Best Trial: (Value: {best_trial.value}, Params: {best_trial.params})')


if __name__ == '__main__':
    # Run the different trials
    run_study(n_trials=TRIALS)

    # Log the ending of the script
    logger.info('Hyperparameter Tuning Succesfully Completed')
    logger.info('-'*80)
