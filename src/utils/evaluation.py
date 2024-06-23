import torch

from numpy import ndarray
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef
)


def get_accuracy(y_true: ndarray, y_pred: ndarray) -> float:
    '''
    Calculate the accuracy of predictions.

    Args:
        y_true (ndarray): The ground truth labels.
        y_pred (ndarray): The predicted labels.

    Returns:
        float: The accuracy score.
    '''
    return accuracy_score(y_true, y_pred)


def get_balanced_accuracy(y_true: ndarray, y_pred: ndarray) -> float:
    '''
    Calculate the balanced accuracy of predictions.

    Args:
        y_true (ndarray): The ground truth labels.
        y_pred (ndarray): The predicted labels.

    Returns:
        float: The balanced accuracy score.
    '''
    return balanced_accuracy_score(y_true, y_pred)


def get_precision(y_true: ndarray, y_pred: ndarray) -> float:
    '''
    Calculate the precision of predictions.

    Args:
        y_true (ndarray): The ground truth labels.
        y_pred (ndarray): The predicted labels.

    Returns:
        float: The precision score.
    '''
    return precision_score(y_true, y_pred)


def get_recall(y_true: ndarray, y_pred: ndarray) -> float:
    '''
    Calculate the recall of predictions.

    Args:
        y_true (ndarray): The ground truth labels.
        y_pred (ndarray): The predicted labels.

    Returns:
        float: The recall score.
    '''
    return recall_score(y_true, y_pred)


def get_specificity(y_true: ndarray, y_pred: ndarray) -> float:
    '''
    Calculate the specificity of predictions.

    Args:
        y_true (ndarray): The ground truth labels.
        y_pred (ndarray): The predicted labels.

    Returns:
        float: The specificity score.
    '''
    tn = torch.sum((y_true == 0) & (y_pred == 0))
    fp = torch.sum((y_true != 0) & (y_pred == 0))
    
    specificity = tn / (tn + fp + 1e-15)  # Adding a small epsilon to avoid division by zero
    
    return specificity.item()


def get_f1_score(y_true: ndarray, y_pred: ndarray) -> float:
    '''
    Calculate the F1 score of predictions.

    Args:
        y_true (ndarray): The ground truth labels.
        y_pred (ndarray): The predicted labels.

    Returns:
        float: The F1 score.
    '''
    return f1_score(y_true, y_pred)


def get_matthews_corrcoef(y_true: ndarray, y_pred: ndarray) -> float:
    '''
    Calculate the Matthews correlation coefficient of predictions.

    Args:
        y_true (ndarray): The ground truth labels.
        y_pred (ndarray): The predicted labels.

    Returns:
        float: The Matthews correlation coefficient.
    '''
    return matthews_corrcoef(y_true, y_pred)


def median_absolute_percentage_error(y_true, y_pred):
    '''
    Calculate the median absolute percentage error of predictions.

    Args:
        y_true (ndarray): The ground truth labels.
        y_pred (ndarray): The predicted labels.

    Returns:
        float: The median absolute percentage error.
    '''
    return torch.median(torch.abs((y_true - y_pred) / (y_true + 1e-15))).item() * 100


def get_confusion_matrix(y_true: ndarray, y_pred: ndarray) -> float:
    '''
    Calculate the confusion matrix of predictions.

    Args:
        y_true (ndarray): The ground truth labels.
        y_pred (ndarray): The predicted labels.

    Returns:
        List[List[int]]: The confusion matrix.
    '''
    return confusion_matrix(y_true, y_pred)
