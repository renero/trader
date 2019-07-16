from keras.layers import Dense, InputLayer
from keras.models import Sequential

from environment import Environment


class NNModel(object):

    def __init__(self, context_dictionary):
        self.__dict__.update(context_dictionary)

    def create_model(self, env: Environment) -> Sequential:
        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1, self._num_states)))
        model.add(
            Dense(self._num_states * self._num_actions, activation='sigmoid'))
        model.add(Dense(self._num_actions, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        model.summary()

        return model
