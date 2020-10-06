import tensorflow as tf


class display_progress(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
        # super().__init__()
        self.epochs = epochs

    def set_params(self, params):
        params['epochs'] = 0

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch+1:03d}/{self.epochs:03d}", end="")

    def on_epoch_end(self, epoch, logs=None):
        print(
            f" - Acc.:{logs['accuracy']:.2f} - Val.Acc.:{logs['val_accuracy']:.02f}")
