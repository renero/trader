from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from nn import nn


class lstm_1layer(nn):

    def __init__(self, params):
        super().__init__(params)

    def build_model(self):
        """
        Builds the model according to the parameters specified for
        the network in the params, or as arguments.
        """
        assert self.params.window_size is not None and \
               self.params.num_features is not None and \
               self.params.num_target_labels is not None, \
            "All parameters to build_model *must* be specified."

        # Build the LSTM
        model = Sequential()
        model.add(
            LSTM(
                input_shape=(self.window_size, self.params.num_features),
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
        model.add(
            Dense(self.params.num_target_labels,
                  activation=self.params.activation))
        model.compile(
            loss=self.params.loss, optimizer=self.params.optimizer,
            metrics=self.params.metrics)

        if self.params.summary is True:
            model.summary()

        self.model = model
        self.metadata['name'] = 'lstm_1layer'
        self.metadata['l1units'] = self.params.l1units
        self.metadata['l2units'] = self.params.l2units

        self.log.debug(
            'NN {} created'.format(self.metadata['name']))

        return self
