from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

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
            self.add_single_lstm_layer(model)
        else:
            for _ in range(self.metadata['layers']):
                self.add_stacked_lstm_layer(model)
            self.add_output_lstm_layer(model)
        if self.metadata['binary'] is True:
            self.close_binary_network(model)
        else:
            self.close_network(model)
        self.model = model

        self.log.info(
            'NN {} created'.format(self.metadata['name']))
        return self


class dense(nn):

    def __init__(self, params, n_layers: int, binary=False):
        super().__init__(params)
        self.metadata['name'] = f'{self.__class__.__name__}_{n_layers}layers'
        self.metadata['layers'] = n_layers
        self.metadata['binary'] = binary
        self._build_model()

    def _build_model(self):
        """
        Builds the model according to the parameters specified for
        the network in the params, or as arguments.
        """
        model = Sequential()
        model.add(Dense(
            50,
            batch_input_shape=(None, self.window_size, self.params.num_features),
            activation='relu'))
        model.add(Dense(
            75,
            activation='relu'))
        n = self.metadata['layers']
        for i in range(n - 1):
            mu = (n - (i + 1)) / n
            model.add(Dense(
                50*mu,
                activation='relu'))
        if self.metadata['binary'] is True:
            self.close_binary_network(model)
        else:
            self.close_network(model)
        self.model = model

        self.log.info(
            'NN {} created'.format(self.metadata['name']))
        return self

# class lstm_1layer(nn):
#
#     def __init__(self, params, binary=False):
#         super().__init__(params)
#         self.metadata['name'] = self.__class__.__name__
#         self.metadata['layers'] = 1
#         self.metadata['binary'] = binary
#         self._build_model(binary)
#
#     def _build_model(self, binary):
#         """
#         Builds the model according to the parameters specified for
#         the network in the params, or as arguments.
#         """
#         model = Sequential()
#         self.add_single_lstm_layer(model)
#         if binary is True:
#             self.close_binary_network(model)
#         else:
#             self.close_network(model)
#         self.model = model
#
#         self.log.info(
#             'NN {} created'.format(self.metadata['name']))
#         return self
#
#
# class lstm_2layer(nn):
#
#     def __init__(self, params, binary=False):
#         super().__init__(params)
#         self.metadata['name'] = self.__class__.__name__
#         self.metadata['layers'] = 2
#         self.metadata['binary'] = binary
#         self._build_model(binary)
#
#     def _build_model(self, binary):
#         """
#         Builds the model according to the parameters specified for
#         the network in the params, or as arguments.
#         """
#         model = Sequential()
#         self.add_stacked_lstm_layer(model)
#         self.add_output_lstm_layer(model)
#         if binary is True:
#             self.close_binary_network(model)
#         else:
#             self.close_network(model)
#         self.model = model
#
#         self.log.info(
#             'NN {} created'.format(self.metadata['name']))
#         return self
