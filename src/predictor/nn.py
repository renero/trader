from collections import defaultdict
from os.path import join, basename, splitext, dirname, realpath
from pathlib import Path

import mlflow
from numpy import ndarray
from tensorflow.keras.models import model_from_json

from metrics import metrics
from utils.callbacks import display_progress
from utils.file_utils import file_exists
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
        self._end_experiment()
        return exp_name

    @staticmethod
    def _set_experiment(name: str) -> str:
        if name is None:
            name = 'untracked'
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

    def _end_experiment(self):
        if self.params.mlflow:
            mlflow.end_run()

    def load(self, model_name, summary=False):
        """ Load json and create model """
        self.log.info('Reading model file: {}'.format(model_name))
        nn_path = join(self.params.models_dir, '{}.json'.format(model_name))
        nn_path = file_exists(nn_path, dirname(realpath(__file__)))
        json_file = open(nn_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        weights_path = join(self.params.models_dir, '{}.h5'.format(model_name))
        weights_path = file_exists(weights_path, dirname(realpath(__file__)))
        loaded_model.load_weights(weights_path)
        loaded_model.compile(
            loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        if summary is True:
            loaded_model.summary()
        self.model = loaded_model

        return loaded_model

    def save(self):
        """ serialize model to JSON """
        if self.metadata['accuracy'] == 'unk':
            raise ValidationException('Trying to save without training.')

        if self.params.output is not None:
            model_name = self.params.output
        else:
            model_name = self._valid_model_name()

        model_json = self.model.to_json()
        with open('{}.json'.format(model_name), "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights('{}.h5'.format(model_name))
        self.log.info("Saved model and weights ({})".format(model_name))

    def _valid_model_name(self):
        """
        Builds a valid name with the metadata and the date.
        Returns The filename if the name is valid and file does not exists,
                None otherwise.
        """
        self.filename = '{}_w{}_e{}'.format(
            self.metadata['dataset'],
            self.metadata['window_size'],
            self.metadata['epochs'])
        base_filepath = join(self.params.models_dir, self.filename)
        output_filepath = base_filepath
        idx = 1
        while Path(output_filepath).is_file() is True:
            output_filepath = '{}_{:d}'.format(base_filepath, idx)
            idx += 1
        return output_filepath
