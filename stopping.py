import math


class EarlyStoppingCallback(object):
    def __init__(self, patience=100):
        #initialize all members you need
        self.patience = patience
        self.patience_count = 0
        self.stop_toggle = False
        self.previous_loss = 1.0
        self.current_loss = None

    def step(self, current_loss):
        # check whether the current loss is lower than the previous best value.
        # if not count up for how long there was no progress
        self.current_loss = current_loss
        self.should_stop()
        return self.stop_toggle

    def should_stop(self):
        # check whether the duration of where there was no progress is larger or equal to the patience
        if (self.current_loss > self.previous_loss) and (self.patience_count == self.patience):
            self.stop_toggle = True
        else:
            self.previous_loss = self.current_loss
            self.patience_count += 1