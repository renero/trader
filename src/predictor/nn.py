import random
from collections import defaultdict
from os.path import basename, splitext

import mlflow
import pandas as pd
import tensorflow as tf
from numpy import ndarray
from pandas import DataFrame
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.regularizers import L2, l2

from metrics import metrics


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

        self.metadata['dataset'] = \
            splitext(basename(self.params.input_file))[0]
        self.metadata['epochs'] = self.params.epochs
        self.metadata['window_size'] = self.params.window_size
        self.metadata['num_features'] = self.params.num_features
        self.metadata['num_target_labels'] = self.params.num_target_labels
        self.metadata['dropout'] = self.params.dropout
        self.metadata['learning_rate'] = self.params.learning_rate
        self.metadata['seed'] = self.params.seed
        self.metadata['activation'] = self.params.activation

        tf.random.set_seed(self.params.seed)

    def start_experiment(self, X_train: ndarray, y_train: ndarray,
                         name=None) -> str:
        exp_name = self._set_experiment(name)
        mlflow.start_run()
        mlflow.keras.autolog()
        self.log.info(f'MLFlow experiment ({exp_name}) started')
        self._train(X_train, y_train)
        return exp_name

    @staticmethod
    def _set_experiment(name: str) -> str:
        if name is None:
            name = f'{random.getrandbits(32):x}'
        mlflow.set_experiment(name)
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
            validation_split=self.params.validation_split)

        mlflow.log_params(nn.metadata)
        return self

    def evaluate_experiment(self, X_test: ndarray,
                            y_test: ndarray) -> DataFrame:
        yhat = self.model.predict(X_test)
        self.log.info(f"Predictions (yhat): {yhat.shape}")

        n_predictions = int(X_test.shape[0])
        Y = y_test.reshape(n_predictions, )
        Yhat = yhat.reshape(n_predictions, )
        result = pd.DataFrame({"y": Y, "yhat": Yhat, }).round(2)

        ta = metrics.trend_accuracy(result)
        mlflow.log_metric("trend_accuracy", ta)
        self.log.info(f"Trend acc.: {ta:4.2f}")

        return result

    @staticmethod
    def end_experiment():
        mlflow.end_run()

    def add_single_lstm_layer(self, model):
        """Single layer LSTM must use this layer."""
        model.add(
            LSTM(
                input_shape=(self.window_size, self.params.num_features),
                dropout=self.params.dropout,
                units=self.params.l1units,
                kernel_regularizer=l2(0.0000001),
                activity_regularizer=l2(0.0000001)))

    def add_output_lstm_layer(self, model):
        """Use this layer, when stacking severeal ones."""
        model.add(
            LSTM(
                self.params.l2units,
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
                units=self.params.l1units,
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

