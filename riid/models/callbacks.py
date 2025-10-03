from tensorflow.keras.callbacks import Callback
from time import perf_counter as timer


class TimeLimitCallback(Callback):
    def __init__(self, max_seconds):
        super().__init__()
        self.max_seconds = max_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = timer()

    def on_epoch_end(self, epoch, logs=None):
        if timer() - self.start_time >= self.max_seconds:
            self.model.stop_training = True
