from src.utils.evaluation import (
    get_accuracy,
    get_balanced_accuracy,
    get_precision,
    get_recall,
    get_specificity,
    get_f1_score,
    get_matthews_corrcoef,
    median_absolute_percentage_error,
    get_confusion_matrix
)
from src.utils.log import configure_logger

import torch
from torch import nn
from torch.utils.data import Dataset

import multiprocessing
from tqdm import tqdm
from typing import Callable, Union, List, Dict


class Evaluator:
    '''
    A class to evaluate a PyTorch model on a test dataset using various metrics.
    '''

    # Initialize the logger as a class attribute
    logger = configure_logger(__name__)

    def __init__(self,
            model: nn.Module,
            test_ds: Dataset,
            cretirion: nn.Module,
            device: torch.device=torch.device('cpu')
        ) -> None:
        '''
        Initializes the Evaluator with the model, test dataset, loss function, and device.

        Args:
            model (nn.Module): The neural network model to be evaluated.
            test_ds (Dataset): Dataset containing the test data.
            criterion (nn.Module): Loss function used for evaluation.
            device (torch.device, optional): Device to run the evaluation on (CPU or GPU). Defaults to CPU.
        '''
        self.model = model.to(device, non_blocking=True)
        self.test_ds = test_ds
        self.cretirion = cretirion
        self.device = device

    def evaluate(self) -> Dict[str, Union[float, List[List[float]]]]:
        '''
        Evaluates the model on the test dataset using various metrics.

        Returns:
            Dict[str, Union[float, List[List[float]]]]: A dictionary containing evaluation metrics.
                - "Loss": The loss value.
                - "accuracy": Accuracy of the model.
                - "balanced_accuracy": Balanced accuracy of the model.
                - "precision": Precision of the model.
                - "recall": Recall of the model.
                - "specificity": Specificity of the model.
                - "f1_score": F1 score of the model.
                - "matthews_corr": Matthews correlation coefficient.
                - "mdape": Median absolute percentage error.
                - "confusion_matrix": Confusion matrix for the predictions.
        '''
        def _initialize_results(manager: multiprocessing.Manager) -> Dict[str, Union[float, List[List[float]]]]:
            return manager.dict({
                'Loss': 0.0,
                'accuracy': 0.0,
                'balanced_accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'specificity': 0.0,
                'f1_score': 0.0,
                'matthews_corr': 0.0,
                'mdape': 0.0,
                'confusion_matrix': [[0.0, 0.0], [0.0, 0.0]] # binary classification problem
            })

        def _define_metrics() -> Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]:
            return {
                'Loss': nn.BCEWithLogitsLoss(),
                'accuracy': get_accuracy,
                'balanced_accuracy': get_balanced_accuracy,
                'precision': get_precision,
                'recall': get_recall,
                'specificity': get_specificity,
                'f1_score': get_f1_score,
                'matthews_corr': get_matthews_corrcoef,
                'mdape': median_absolute_percentage_error,
                'confusion_matrix': get_confusion_matrix
            }

        def _calculate_and_update(
                y_true: torch.Tensor,
                y_pred: torch.Tensor,
                key: str,
                metric: Callable[[torch.Tensor, torch.Tensor], Union[List[float], float]],
            ) -> None:
            '''
            Calculate the specified metric and update the results dictionary.

            Args:
                y_true (torch.Tensor): The ground truth labels.
                y_pred (torch.Tensor): The predicted labels.
                key (str): The metric name.
                metric (Callable[[torch.Tensor, torch.Tensor], Union[List[float], float]]): The metric function.
            '''
            nonlocal results
            if key == 'Loss':
                results[key] = metric(y_pred, y_true).item()
            else:
                results[key] = metric(y_true, y_pred)

        manager = multiprocessing.Manager()
        results = _initialize_results(manager)
        metrics = _define_metrics()

        Evaluator.logger.info('Start Evaluation Process.')

        y_pred = []
        y_true = []

        self.model.eval()
        with torch.inference_mode():
            for x_test, y_test in tqdm(self.test_ds, ascii=True, desc='    Producing Predictions'):
                # Move samples to the same device as the model
                x_test = x_test.to(self.device, non_blocking=True)

                y_logits = self.model(x_test.unsqueeze(dim=0))

                y_pred.append(torch.round(torch.sigmoid(y_logits.reshape(-1))).item())
                y_true.append(y_test)

        # Start multiprocessing for metric calculations
        processes = []
        for metric_name, metric_fn in tqdm(metrics.items(), ascii=True, desc='    Calculating Metrics'):
            process = multiprocessing.Process(target=_calculate_and_update, args=(torch.tensor(y_true), torch.tensor(y_pred), metric_name, metric_fn))
            processes.append(process)
            process.start()

        # Ensure all processes have completed
        for process in processes:
            process.join()

        # Convert manager list back to a normal list for the confusion matrix
        results['confusion_matrix'] = list(results['confusion_matrix'])

        Evaluator.logger.info("Evaluation Process Completed Successfully.")

        return dict(results)
