import random
from collections import defaultdict
from os.path import basename, splitext

import mlflow
from numpy import ndarray
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from metrics import metrics
from utils.utils import reset_seeds
from utils.callbacks import display_progress



class ValidationException(Exception):
    pass


class nn:
    metadata = defaultdict(str)
    output_dir = ''
    history = None
    window_size = None
    num_categories = None
    model = None
    yhat = None
    filename = None

    def __init__(self, params):
        """
        Init the class with the number of categories used to encode candles
        """
        self.params = params
        self.log = params.log

        reset_seeds()

        self.metadata['dataset'] = \
            splitext(basename(self.params.input_file))[0]
        self.metadata['epochs'] = self.params.epochs
        self.metadata['window_size'] = self.params.window_size
        self.metadata['num_features'] = self.params.num_features
        self.metadata['num_target_labels'] = self.params.num_target_labels
        self.metadata['dropout'] = self.params.dropout
        self.metadata['learning_rate'] = self.params.learning_rate
        self.metadata['activation'] = self.params.activation
        self.metadata['units'] = self.params.units
        self.metadata['batch_size'] = self.params.batch_size

    def start_training(self, X_train: ndarray, y_train: ndarray,
                       name=None) -> str:

        self.log.info(f'Training for {self.params.epochs} epochs...')
        exp_name = self._set_experiment(name) if self.params.mlflow else None
        self._train(X_train, y_train)
        return exp_name

    @staticmethod
    def _set_experiment(name: str) -> str:
        if name is None:
            name = f'{random.getrandbits(32):x}'
        mlflow.set_experiment(name)
        mlflow.start_run()
        mlflow.keras.autolog()
        return name

    def _train(self, X_train: ndarray, y_train: ndarray) -> "nn":
        """
        Train the model and put the history in an internal state.
        Metadata is updated with the accuracy
        """
        assert self.model is not None, "model not yet build."
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=self.params.epochs,
            batch_size=self.params.batch_size,
            verbose=self.params.verbose,
            validation_split=self.params.validation_split,
            callbacks=[display_progress(self.params.epochs)]
        )

        if self.params.mlflow:
            mlflow.log_params(nn.metadata)

        return self

    def evaluate(self, X_test: ndarray,
                 y_test: ndarray) -> (ndarray, float):
        # Get predictions and also evaluate the model
        yhat = self.model.predict(X_test)
        _, accuracy = self.model.evaluate(X_test, y_test)
        self.log.info(f"Predictions (yhat): {yhat.shape}")
        self.log.info(f"Accuracy: {accuracy:.2f}")

        if self.metadata['binary'] is True:
            ta = metrics.trend_binary_accuracy(y_test, yhat)
        else:
            ta = metrics.trend_accuracy(y_test, yhat)
        if self.params.mlflow:
            mlflow.log_metric("trend_accuracy", ta)
        self.log.info(f"Trend acc.: {ta:4.2f}")

        return yhat, ta

    def end_experiment(self):
        if self.params.mlflow:
            mlflow.end_run()

    def add_single_lstm_layer(self, model):
        """Single layer LSTM must use this layer."""
        model.add(
            LSTM(
                input_shape=(self.window_size, self.params.num_features),
                dropout=self.params.dropout,
                units=self.params.units,
                kernel_regularizer=l2(0.0000001),
                activity_regularizer=l2(0.0000001)))

    def add_output_lstm_layer(self, model):
        """Use this layer, when stacking several ones."""
        model.add(
            LSTM(
                self.params.units,
                dropout=self.params.dropout,
                kernel_regularizer=l2(0.0000001),
                activity_regularizer=l2(0.0000001)))

    def add_stacked_lstm_layer(self, model):
        """Use N-1 of this layer to stack N LSTM layers."""
        model.add(
            LSTM(
                input_shape=(self.window_size, self.params.num_features),
                return_sequences=True,
                dropout=self.params.dropout,
                units=self.params.units,
                kernel_regularizer=l2(0.0000001),
                activity_regularizer=l2(0.0000001)))

    def close_network(self, model):
        """Adds a dense layer, and compiles the model with the selected
        optimzer, returning a summary of the model, if set to True in params"""
        model.add(
            Dense(self.params.num_target_labels,
                  activation=self.params.activation))
        optimizer = Adam(lr=self.params.learning_rate)
        model.compile(
            loss=self.params.loss,
            optimizer=optimizer,
            metrics=self.params.metrics)

        if self.params.summary is True:
            model.summary()

    def close_binary_network(self, model):
        """ Last layer to predict binary outputs """
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(lr=self.params.learning_rate)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics='accuracy')

        if self.params.summary is True:
            model.summary()
