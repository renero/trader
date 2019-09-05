import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, model_from_json
from keras.regularizers import l2
from os.path import join, basename, splitext
from pathlib import Path

from params import Params


class ValidationException(Exception):
    pass


class Csnn(Params):
    _num_categories = 0
    _window_size = 3
    _num_predictions = 1
    _test_size = 0.1
    _dropout = 0.1
    _history = None
    _enc_data = None
    _raw_data = None
    _input_file = ''

    # metadata
    _metadata = {'period': 'unk', 'epochs': 'unk', 'accuracy': 'unk'}

    # Files output
    _output_dir = ''

    # Model design
    _l1units = 256
    _l2units = 256
    _activation = 'sigmoid'
    _model = None

    # Training
    _epochs = 100
    _batch_size = 10
    _validation_split = 0.1
    _verbose = 1

    # Compilation
    _loss = 'mean_squared_error'
    _optimizer = 'adam'
    _metrics = ['accuracy']

    # Results
    _history = None

    def __init__(self, name, subtype):
        """
        Init the class with the number of categories used to encode candles
        """
        super(Csnn, self).__init__()
        self._metadata['dataset'] = splitext(basename(self._ticks_file))[0]
        self._metadata['epochs'] = self._epochs
        if name is not None:
            self.name = name
        else:
            self.name = self._metadata['dataset']
        self.subtype = subtype
        self._metadata['subtype'] = subtype
        self.log.info(
            'NN {}.{} created'.format(self.name, self._metadata['subtype']))

    def build_model(self, window_size=None, num_categories=None, summary=True):
        """
        Builds the model according to the parameters specified for
        the network in the params, or as arguments.
        """
        # Override, if necessary, the input and window sizes with the values
        # passed as arguments.
        if window_size is not None:
            self._window_size = window_size
        if num_categories is not None:
            self._num_categories = num_categories

        # Build the LSTM
        model = Sequential()
        model.add(
            LSTM(
                input_shape=(self._window_size, self._num_categories),
                return_sequences=True,
                units=self._l1units,
                kernel_regularizer=l2(0.0000001),
                activity_regularizer=l2(0.0000001)))
        model.add(Dropout(self._dropout))
        model.add(
            LSTM(
                self._l2units,
                kernel_regularizer=l2(0.0000001),
                activity_regularizer=l2(0.0000001)))
        model.add(Dropout(self._dropout))
        model.add(Dense(self._num_categories, activation=self._activation))
        model.compile(
            loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)
        if summary is True:
            model.summary()
        self._model = model
        return self

    def train(self, X_train, y_train):
        """
        Train the model and put the history in an internal state.
        Metadata is updated with the accuracy
        """
        self._history = self._model.fit(
            X_train,
            y_train,
            epochs=self._epochs,
            batch_size=self._batch_size,
            verbose=self._verbose,
            validation_split=self._validation_split)
        self._metadata[self._metrics[0]] = self._history.history['acc']
        return self

    def predict(self, test_set):
        """
        Make a prediction over the internal X_test set.
        """
        info_msg = 'Network {}/{} making prediction'
        self.log.info(info_msg.format(self.name, self.subtype))
        self._yhat = self._model.predict(test_set)
        return self._yhat

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
        self._filename = '{}_{}_{}_w{}_e{}_a{:.4f}'.format(
            self._metadata['subtype'],
            datetime.now().strftime('%Y%m%d_%H%M'),
            self._metadata['dataset'],
            self._window_size,
            self._epochs,
            self._metadata['accuracy'][-1])
        base_filepath = join(self._models_dir, self._filename)
        output_filepath = base_filepath
        idx = 1
        while Path(output_filepath).is_file() is True:
            output_filepath = '{}_{:d}'.format(base_filepath + idx)
            idx += 1
        return output_filepath

    def load(self, model_name, summary=False):
        """ Load json and create model """
        self.log.info('Reading model file: {}'.format(model_name))
        json_file = open(
            join(self._models_dir, '{}.json'.format(model_name)), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(
            join(self._models_dir, '{}.h5'.format(model_name)))
        loaded_model.compile(
            loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        if summary is True:
            loaded_model.summary()
        self._model = loaded_model

        return loaded_model

    def save(self, modelname=None):
        """ serialize model to JSON """
        if self._metadata['accuracy'] == 'unk':
            raise ValidationException('Trying to save without training.')
        if modelname is None:
            modelname = self.valid_output_name()
        model_json = self._model.to_json()
        with open('{}.json'.format(modelname), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self._model.save_weights('{}.h5'.format(modelname))
        self.log.info("Saved model and weights to disk")

    # TODO: Take this function out to the CSPlot class.
    def plot_history(self):
        if self._history is None:
            raise ValidationException('Trying to plot without training')
        """ summarize history for accuracy and loss """
        plt.plot(self._history.history['acc'])
        plt.plot(self._history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(self._history.history['loss'])
        plt.plot(self._history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
