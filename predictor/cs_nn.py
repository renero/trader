from os.path import join, basename, splitext, dirname, realpath
from pathlib import Path

import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, model_from_json
from keras.regularizers import l2

from utils.plots import plot_history
from utils.file_io import file_exists


class ValidationException(Exception):
    pass


class CS_NN(object):

    metadata = {'period': 'unk', 'epochs': 'unk', 'accuracy': 'unk'}
    output_dir = ''
    history = None
    window_size = None
    num_categories = None
    model = None
    yhat = None
    filename = None

    def __init__(self, params, name, subtype):
        """
        Init the class with the number of categories used to encode candles
        """
        super(CS_NN, self).__init__()
        self.params = params
        self.log = params.log

        self.metadata['dataset'] = splitext(basename(self.params.input_file))[0]
        self.metadata['epochs'] = self.params.epochs
        if name is not None:
            self.name = name
        else:
            self.name = self.metadata['dataset']
        self.subtype = subtype
        self.metadata['subtype'] = subtype
        self.log.debug(
            'NN {}.{} created'.format(self.name, self.metadata['subtype']))

    def build_model(self, window_size=None, num_categories=None, summary=True):
        """
        Builds the model according to the parameters specified for
        the network in the params, or as arguments.
        """
        # Override, if necessary, the input and window sizes with the values
        # passed as arguments.
        if window_size is not None:
            self.window_size = window_size
        if num_categories is not None:
            self.num_categories = num_categories

        # Build the LSTM
        model = Sequential()
        model.add(
            LSTM(
                input_shape=(self.window_size, self.num_categories),
                return_sequences=True,
                units=self.params.l1units,
                kernel_regularizer=l2(0.0000001),
                activity_regularizer=l2(0.0000001)))
        model.add(Dropout(self.params.dropout))
        model.add(
            LSTM(
                self.params.l2units,
                kernel_regularizer=l2(0.0000001),
                activity_regularizer=l2(0.0000001)))
        model.add(Dropout(self.params.dropout))
        model.add(Dense(self.num_categories, activation=self.params.activation))
        model.compile(
            loss=self.params.loss, optimizer=self.params.optimizer,
            metrics=self.params.metrics)
        if summary is True:
            model.summary()
        self.model = model
        return self

    def train(self, X_train, y_train):
        """
        Train the model and put the history in an internal state.
        Metadata is updated with the accuracy
        """
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=self.params.epochs,
            batch_size=self.params.batch_size,
            verbose=self.params.verbose,
            validation_split=self.params.validation_split)
        self.metadata[self.params.metrics[0]] = self.history.history['acc']

        if self.params.plot is True:
            plot_history(self.history)

        return self

    def predict(self, test_set):
        """
        Make a prediction over the internal X_test set.
        """
        info_msg = 'Network {}/{} making prediction'
        self.log.debug(info_msg.format(self.name, self.subtype))
        self.yhat = self.model.predict(test_set)
        return self.yhat

    def hardmax(self, y):
        """From the output of a tanh layer, this method makes every position
        in the vector equal to zero, except the largest one, which is valued
        -1 or +1.

        Argument:
          - A numpy vector of predictions of shape (1, p) with values in
            the range (-1, 1)
        Returns:
          - A numpy vector of predictions of shape (1, p) with all elements
            in the vector equal to 0, except the max one, which is -1 or +1
        """
        self.log.debug('Getting hardmax of: {}'.format(y))
        min = np.argmin(y)
        max = np.argmax(y)
        pos = max if abs(y[max]) > abs(y[min]) else min
        y_max = np.zeros(y.shape)
        y_max[pos] = 1.0 if pos == max else -1.0
        self.log.debug('Hardmax position = {}'.format(y_max))
        return y_max

    def valid_output_name(self):
        """
        Builds a valid name with the metadata and the date.
        Returns The filename if the name is valid and file does not exists,
                None otherwise.
        """
        self.filename = '{}_{}_w{}_e{}'.format(
            self.metadata['subtype'],
            self.metadata['dataset'],
            self.window_size,
            self.params.epochs)
        base_filepath = join(self.params.models_dir, self.filename)
        output_filepath = base_filepath
        idx = 1
        while Path(output_filepath).is_file() is True:
            output_filepath = '{}_{:d}'.format(base_filepath, idx)
            idx += 1
        return output_filepath

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

    def save(self, modelname=None):
        """ serialize model to JSON """
        if self.metadata['accuracy'] == 'unk':
            raise ValidationException('Trying to save without training.')
        if modelname is None:
            modelname = self.valid_output_name()
        model_json = self.model.to_json()
        with open('{}.json'.format(modelname), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights('{}.h5'.format(modelname))
        self.log.info("Saved model and weights ({})".format(modelname))
