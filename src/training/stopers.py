class EarlyStopping:
    '''
    EarlyStopping class to stop training when a monitored metric stops improving.
    '''

    def __init__(self,
            patience: int=5,
            threshold: float=0,
        ) -> None:
        '''
        Initializes the EarlyStopping instance with the given patience and threshold.

        Parameters:
        ----------
        patience : int, optional (default=5)
            Number of epochs to wait for improvement before stopping the training.
        
        threshold : float, optional (default=0)
            Minimum change in the monitored metric to qualify as an improvement.
        '''
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False 

    def __call__(self, valid_loss: float) -> None:
        '''
        Checks if the validation loss has improved and updates the early stopping criteria.

        Parameters:
        ----------
        valid_loss : float
            The current validation loss.

        Returns:
        -------
        bool
            Flag indicating whether training should be stopped.
        '''
        if valid_loss > self.best_score * (1 + self.threshold):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = valid_loss
            self.counter = 0

        return self.early_stop

    def reset(self) -> None:
        '''
        Resets the early stopping criteria. This can be useful if the early stopping 
        needs to be reused for another training session.
        '''
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False 
