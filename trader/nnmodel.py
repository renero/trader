from keras.layers import Dense, InputLayer
from keras.models import Sequential

from environment import Environment


class NNModel:

    @staticmethod
    def create_model(env: Environment) -> Sequential:
        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1, env.num_states_)))
        model.add(
            Dense(env.num_states_ * env.num_actions_, activation='sigmoid'))
        model.add(Dense(env.num_actions_, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        model.summary()

        return model
