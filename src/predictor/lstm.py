from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from nn import nn


class lstm_1layer(nn):

    def __init__(self, params, binary=False):
        super().__init__(params)
        self.metadata['name'] = self.__class__.__name__
        self.metadata['layers'] = 1
        self.metadata['binary'] = binary
        self._build_model(binary)

    def _build_model(self, binary):
        """
        Builds the model according to the parameters specified for
        the network in the params, or as arguments.
        """
        model = Sequential()
        self.add_single_lstm_layer(model)
        if binary is True:
            self.close_binary_network(model)
        else:
            self.close_network(model)
        self.model = model

        self.log.info(
            'NN {} created'.format(self.metadata['name']))
        return self


class lstm_2layer(nn):

    def __init__(self, params, binary=False):
        super().__init__(params)
        self.metadata['name'] = self.__class__.__name__
        self.metadata['layers'] = 2
        self.metadata['binary'] = binary
        self._build_model(binary)

    def _build_model(self, binary):
        """
        Builds the model according to the parameters specified for
        the network in the params, or as arguments.
        """
        model = Sequential()
        self.add_stacked_lstm_layer(model)
        self.add_output_lstm_layer(model)
        if binary is True:
            self.close_binary_network(model)
        else:
            self.close_network(model)
        self.model = model

        self.log.info(
            'NN {} created'.format(self.metadata['name']))
        return self

