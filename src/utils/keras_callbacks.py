import tensorflow as tf

from strings import print_progbar


class display_progress(tf.keras.callbacks.Callback):
    blades = ['|', '/', '–', '\\']
    pos = 0

    def __init__(self, epochs):
        self.epochs = epochs
        self.a_max = -1000.0
        self.a_min = 1000.
        self.v_max = -1000.0
        self.v_min = 1000.

        self.refresh_step = 10 if self.epochs > 20 else 1

    def set_params(self, params):
        params['epochs'] = 0

    def rotate_blades(self):
        print('\r' + self.blades[self.pos], sep="", end="")
        self.pos = self.pos + 1 if self.pos < len(self.blades) - 1 else 0

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        acc, v_acc = self.get_min_and_amx(logs)

        if epoch % self.refresh_step != 0:
            return

        str_epoch = f"\rEpoch {epoch:03d}/{self.epochs:03d}"
        str_acc = f" - Acc:{acc:.2f} (↑{self.a_max:.02f}/↓{self.a_min:.02f})"
        str_val = f" - Val:{v_acc:.02f} (↑{self.v_max:.02f}/↓{self.v_min:.02f})"
        pb = print_progbar(epoch/self.epochs, do_print=False)
        print("\r" + str_epoch + str_acc + str_val + ' | ' + pb, end="")

    def get_min_and_amx(self, logs):
        acc = logs['accuracy']
        self.a_max = acc if acc > self.a_max else self.a_max
        self.a_min = acc if acc < self.a_min else self.a_min
        v_acc = logs['val_accuracy']
        self.v_max = v_acc if v_acc > self.v_max else self.v_max
        self.v_min = v_acc if v_acc < self.v_min else self.v_min
        return acc, v_acc

    def on_train_end(self, logs=None):
        print()

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass
