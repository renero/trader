from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from nn import nn


class lstm(nn):

    def __init__(self, params, n_layers: int, binary=False):
        super().__init__(params)
        self.metadata['name'] = f'{self.__class__.__name__}_{n_layers}layers'
        self.metadata['layers'] = n_layers
        self.metadata['binary'] = binary
        self._build_model()

    def __str__(self):
        l = self.metadata['layers']
        b = 'b' if self.metadata['binary'] is True else ''
        d = self.metadata['dropout']
        w = self.metadata['window_size']
        u = self.metadata['units']
        bs = self.metadata['batch_size']
        a = self.metadata['learning_rate']
        e = self.metadata['epochs']
        desc = f"LSTM{b}({l}l. {u}u. d={d:.2f} lr={a} [W={w} E={e} BS={bs}])"
        return desc

    def _build_model(self):
        """
        Builds the model according to the parameters specified for
        the network in the params, or as arguments.
        """
        model = Sequential()
        if self.metadata['layers'] == 1:
            self.add_single_lstm_layer(self.window_size, self.params, model)
        else:
            for _ in range(self.metadata['layers']):
                self.add_stacked_lstm_layer(self.window_size, self.params, model)
            self.add_output_lstm_layer(self.params, model)
        if self.metadata['binary'] is True:
            self.close_binary_network(self.params, model)
        else:
            self.close_lstm_network(self.params, model)
        self.model = model

        self.log.info(
            'NN {} created'.format(self.metadata['name']))
        return self

    @staticmethod
    def add_single_lstm_layer(window_size, params, model):
        """Single layer LSTM must use this layer."""
        model.add(
            LSTM(
                input_shape=(window_size, params.num_features),
                dropout=params.dropout,
                units=params.units,
                kernel_regularizer=l2(0.0000001),
                activity_regularizer=l2(0.0000001)))

    @staticmethod
    def add_output_lstm_layer(params, model):
        """Use this layer, when stacking several ones."""
        model.add(
            LSTM(
                params.units,
                dropout=params.dropout,
                kernel_regularizer=l2(0.0000001),
                activity_regularizer=l2(0.0000001)))

    @staticmethod
    def add_stacked_lstm_layer(window_size, params, model):
        """Use N-1 of this layer to stack N LSTM layers."""
        model.add(
            LSTM(
                input_shape=(window_size, params.num_features),
                return_sequences=True,
                dropout=params.dropout,
                units=params.units,
                kernel_regularizer=l2(0.0000001),
                activity_regularizer=l2(0.0000001)))

    @staticmethod
    def close_lstm_network(params, model):
        """Adds a dense layer, and compiles the model with the selected
        optimzer, returning a summary of the model, if set to True in params"""
        model.add(
            Dense(params.num_target_labels,
                  activation=params.activation))
        optimizer = Adam(lr=params.learning_rate)
        model.compile(
            loss=params.loss,
            optimizer=optimizer,
            metrics=params.metrics)

        if params.summary is True:
            model.summary()

    @staticmethod
    def close_binary_network(params, model):
        """ Last layer to predict binary outputs """
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(lr=params.learning_rate)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics='accuracy')

        if params.summary is True:
            model.summary()
