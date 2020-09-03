from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from nn import nn


class lstm_1layer(nn):

    def __init__(self, params):
        super().__init__(params)
        self._build_model()

    def _build_model(self):
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
                dropout=self.params.dropout,
                units=self.params.l1units,
                kernel_regularizer=l2(0.0000001),
                activity_regularizer=l2(0.0000001)))
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

        self.log.info('Model built.')
        self.model = model
        self.metadata['name'] = 'lstm_1layer'
        self.metadata['layers'] = 1
        self.metadata['l1units'] = self.params.l1units
        self.log.info('Metadata Updated.')

        self.log.info(
            'NN {} created'.format(self.metadata['name']))

        return self
