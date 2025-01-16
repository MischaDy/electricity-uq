"""
based on: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
"""


class EarlyStopper:
    def __init__(self, patience: int, min_delta=0):
        assert patience >= 0
        assert min_delta >= 0
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.done = False

    def should_stop(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            self.done = self.counter > self.patience
        return self.done
