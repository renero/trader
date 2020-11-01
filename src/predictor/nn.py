import json
from collections import defaultdict
from os.path import join, basename, splitext, dirname, realpath

import mlflow
import numpy as np
from numpy import ndarray
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam

from metrics import metrics
from utils.callbacks import display_progress
from utils.file_utils import file_exists, valid_output_name
from utils.utils import reset_seeds


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
        if 'num_features' in self.params:
            self.metadata['num_features'] = self.params.num_features
        if 'num_target_labels' in self.params:
            self.metadata['num_target_labels'] = self.params.num_target_labels
        if 'num_target_values' in self.params:
            self.metadata['num_target_values'] = self.params.num_target_values
        if 'output_values' in self.params:
            self.metadata['output_mapping'] = self.params.output_values
        self.metadata['dropout'] = self.params.dropout
        self.metadata['learning_rate'] = self.params.learning_rate
        self.metadata['activation'] = self.params.activation
        self.metadata['units'] = self.params.units
        self.metadata['batch_size'] = self.params.batch_size
        if 'train_columns' in self.params:
            self.metadata['train_columns'] = self.params.train_columns
        if 'predict_column' in self.params:
            self.metadata['predict_column'] = self.params.predict_column

    def __str__(self):
        p = self.metadata['predict_column']
        l = self.metadata['layers']
        b = 'b' if self.metadata['binary'] is True else ''
        d = self.metadata['dropout']
        w = self.metadata['window_size']
        u = self.metadata['units']
        bs = self.metadata['batch_size']
        a = self.metadata['learning_rate']
        e = self.metadata['epochs']
        desc = f"LSTM{b}_{p};{l}L;u{u};d{d:.2f};lr{a};W{w};E{e};BS{bs}"
        return desc

    def start_training(self, X_train: ndarray, y_train: ndarray,
                       name=None) -> str:

        self.log.info(f'Training for {self.params.epochs} epochs...')
        exp_name = self._set_experiment(name) if self.params.mlflow else None
        self._train(X_train, y_train)
        self._end_experiment()
        return exp_name

    @staticmethod
    def _set_experiment(name: str) -> str:
        if name is None:
            name = 'untracked'
        mlflow.set_experiment(name)
        try:
            mlflow.end_run()
        except Exception:
            pass
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
            callbacks=[display_progress(self.params)]
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

        return yhat, accuracy

    def _end_experiment(self):
        if self.params.mlflow:
            mlflow.end_run()

    #
    # LOAD AND SAVE MODEL FILES
    #

    @staticmethod
    def _load(model_name, params):
        """ Load json and h5 files, compile the model and return it, together
        with the metadata """
        loaded_model = nn._load_model(model_name, params)
        nn._load_weights(loaded_model, model_name, params)
        nn._compile_model(loaded_model, params)
        metadata = nn._load_metadata(model_name, params)

        if params.summary is True:
            loaded_model.summary()
        return loaded_model, metadata

    @staticmethod
    def _compile_model(loaded_model, params):
        optimizer = Adam(lr=params.learning_rate)
        loaded_model.compile(
            loss=params.loss,
            optimizer=optimizer,
            metrics=params.metrics)

    @staticmethod
    def _load_weights(loaded_model, model_name, params):
        # load weights into new model
        weights_path = join(params.models_dir, '{}.h5'.format(model_name))
        weights_path = file_exists(weights_path, dirname(realpath(__file__)))
        loaded_model.load_weights(weights_path)

    @staticmethod
    def _load_model(model_name, params):
        nn_path = join(params.models_dir, '{}.json'.format(model_name))
        nn_path = file_exists(nn_path, dirname(realpath(__file__)))
        json_file = open(nn_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        return loaded_model

    @staticmethod
    def _load_metadata(model_name, params):
        nn_path = join(params.models_dir, '{}.metadata.json'.format(model_name))
        nn_path = file_exists(nn_path, dirname(realpath(__file__)))
        json_file = open(nn_path, 'r')
        loaded_metadata = json.load(json_file)
        json_file.close()
        return loaded_metadata

    def save(self):
        """ serialize model to JSON """
        model_name = valid_output_name(str(self), self.params.models_dir)
        self.log.info(model_name)

        # Save the metadata
        meta_json = json.dumps(self.metadata, sort_keys=True, indent=4)
        with open(f'{model_name}.metadata.json', "w") as json_file:
            json_file.write(meta_json)
        json_file.close()

        # Save the model structure in JSON file (Keras)
        model_json = self.model.to_json()
        with open(f'{model_name}.json', "w") as json_file:
            json_file.write(model_json)
        json_file.close()

        # Serialize weights to HDF5 (Keras)
        self.model.save_weights(f'{model_name}.h5')
        self.log.info("Saved model and weights ({})".format(model_name))
