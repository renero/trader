from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from nn import nn


class lstm_1layer(nn):

    def __init__(self, params):
        super().__init__(params)
        self.metadata['name'] = self.__class__.__name__
        self.metadata['layers'] = 1
        self._build_model()

    def _build_model(self):
        """
        Builds the model according to the parameters specified for
        the network in the params, or as arguments.
        """
        model = Sequential()
        self.add_single_lstm_layer(model)
        self.close_network(model)
        self.model = model

        self.log.info(
            'NN {} created'.format(self.metadata['name']))
        return self


class lstm_2layer(nn):

    def __init__(self, params):
        super().__init__(params)
        self.metadata['name'] = self.__class__.__name__
        self.metadata['layers'] = 2
        self._build_model()

    def _build_model(self):
        """
        Builds the model according to the parameters specified for
        the network in the params, or as arguments.
        """
        model = Sequential()
        self.add_stacked_lstm_layer(model)
        self.add_output_lstm_layer(model)
        self.close_network(model)
        self.model = model

        self.log.info(
            'NN {} created'.format(self.metadata['name']))
        return self

