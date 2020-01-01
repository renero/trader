import os
import random
from os.path import splitext, basename

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Dense, InputLayer
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam

from file_io import valid_output_name
from utils.dictionary import Dictionary

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class RL_NN:
    model = None

    def __init__(self, configuration: Dictionary):
        self.params = configuration
        self.log = self.params.log
        self.model = None

        self.callback_args = {}
        if self.params.tensorboard is True:
            self.log.info('Using TensorBoard')
            self.tensorboard = TensorBoard(
                log_dir=self.params.tbdir,
                histogram_freq=0, write_graph=True, write_images=False)
            self.callback_args = {'callbacks': self.tensorboard}

    def create_model(self) -> Sequential:
        self.model = Sequential()

        # Input layer
        self.model.add(
            InputLayer(batch_input_shape=(None, self.params.num_state_bits)))
        first_layer = True

        # Create all the layers
        for num_cells in self.params.deep_qnet.hidden_layers:
            if first_layer:
                self.model.add(Dense(
                    num_cells,
                    input_shape=(self.params.num_state_bits,),
                    activation=self.params.deep_qnet.activation))
                first_layer = False
            else:
                self.model.add(Dense(num_cells,
                                     activation=self.params.deep_qnet.activation))

        # Output Layer
        last_layer_cells = self.params.deep_qnet.hidden_layers[-1]
        self.model.add(
            Dense(self.params.num_actions, input_shape=(last_layer_cells,),
                  activation='linear'))
        self.compile_model()
        if self.params.debug and self.params.log_level > 2:
            self.log.debug('Model Summary')
            self.model.summary()

        return self.model

    def compile_model(self):
        self.model.compile(
            loss=self.params.deep_qnet.loss,
            optimizer=Adam(lr=self.params.deep_qnet.lr),
            metrics=self.params.deep_qnet.metrics)
        return self.model

    def do_learn(self, episode, episode_step, memory) -> (float, float):
        """ perform minibatch learning or experience replay """
        self.log.debug('Time to learn')
        loss = 0.
        mae = 0.
        if self.params.experience_replay is True:
            loss, mae = self.experience_replay(memory)
        else:
            loss, mae = self.minibatch_learn(memory)
        return loss, mae

    def minibatch_learn(self, memory):
        """
        MiniBatch Learning routine.
        :param memory:
        :return: loss and mae
        """
        mem_size = len(memory)
        if mem_size < self.params.batch_size:
            self.log.debug('Not enough samples for minibatch learn, skipping')
            return 0.0, 0.0

        self.log.debug('Minibatch learn')
        mini_batch = np.empty(shape=(0, 5), dtype=np.int32)
        for i in range(mem_size - self.params.batch_size - 1, mem_size - 1):
            mini_batch = np.append(
                mini_batch,
                np.asarray(memory[i]).astype(int).reshape(1, -1),
                axis=0)
        nn_input, nn_output = self.prepare_nn_data(mini_batch)
        h = self.model.fit(
            nn_input, nn_output,
            epochs=1, verbose=0, batch_size=self.params.batch_size,
            **self.callback_args)
        return h.history['loss'][0], h.history['mae'][0]

    def experience_replay(self, memory):
        """
        Primarily from: https://github.com/edwardhdlu/q-trader
        :param memory:
        :return: loss and mae.
        """
        if len(memory) <= self.params.exp_batch_size:
            self.log.debug('  Not enough samples in experience memory')
            return 0., 0.

        mini_batch = random.sample(memory,
                                   self.params.exp_batch_size)
        nn_input, nn_output = self.prepare_nn_data(mini_batch)
        h = self.model.fit(
            nn_input, nn_output,
            epochs=1, verbose=0, batch_size=self.params.exp_batch_size,
            **self.callback_args)
        self.log.debug('  Learnt exp. batch, loss/mae: {:.2f}/{:.2f}'.format(
            h.history['loss'][0], h.history['mae'][0]))
        return h.history['loss'][0], h.history['mae'][0]

    def prepare_nn_data(self, mini_batch):
        """
        Shape the input and output (supervised labels) to the network from a
        minibatch o previous experiences.
        :param mini_batch:  array of tuples with states, actions, rewards,
                            next states and done values.
        :return: input and output to the network.
        """
        nn_input = np.empty((0, self.params.num_state_bits), dtype=np.int32)
        nn_output = np.empty((0, self.params.num_actions))
        for state, action, reward, next_state, done in mini_batch:
            y = self.prepare_nn_output(state, action, reward, next_state, done)
            nn_input = np.append(nn_input, self.onehot(state), axis=0)
            nn_output = np.append(nn_output, y, axis=0)
        return nn_input, nn_output

    def prepare_nn_output(self, state, action, reward, next_state, done):
        """
        Feed forward the current state to the network to get the output and
        reformat it to set the reward associated to the action taken and thus
        learn that state -> reward association for that action.
        :param state: the state
        :param action: action
        :param reward: reward
        :param next_state: next state
        :param done: if loop has ended
        :return: an array of `n` values, where `n` is the nr of actions.
        """
        target = reward
        if not done:
            target = reward + self.params.gamma * self.predict_value(
                next_state)
        labeled_output = self.model.predict(self.onehot(state))[0]
        labeled_output[action] = target
        y = labeled_output.reshape(-1, self.params.num_actions)
        return y

    def infer_strategy(self) -> list:
        """
        Get the defined strategy from the weights of the model.
        :return: strategy matrix
        """
        strategy = [
            np.argmax(
                self.model.predict(self.onehot(i))[0])
            for i in range(self.params.num_states)
        ]
        return strategy

    def onehot(self, state: int):
        bits_formatter = '{{:0{}b}}'.format(int(self.params.num_state_bits))
        bits = bits_formatter.format(state)
        encoded = list(map(lambda x: -1 if x == '0' else 1, bits))
        return np.array(encoded).reshape(1, self.params.num_state_bits)

    def predict(self, state) -> int:
        return int(np.argmax(self.model.predict(self.onehot(state))))

    def predict_value(self, state):
        return np.max(self.model.predict(self.onehot(state)))

    #
    # Saving and loading the model
    #

    def save_model(self, model, results):
        self.log.info('Saving model, weights and results.')

        if self.params.output is not None:
            fname = self.params.output
        else:
            fname = 'rl_model_' + splitext(
                basename(self.params.forecast_file))[0]
        model_name = valid_output_name(fname, self.params.models_dir, 'json')

        # serialize model to JSON
        model_json = model.to_json()
        with open(model_name, 'w') as json_file:
            json_file.write(model_json)
        self.log.info('  Model: {}'.format(model_name))

        # Serialize weights to HDF5
        weights_name = model_name.replace('.json', '.h5')
        model.save_weights(weights_name)
        self.log.info('  Weights: {}'.format(weights_name))

        # Save also the results table
        results_name = model_name.replace('.json', '.csv')
        results.to_csv(results_name,
                       sep=',',
                       index=False,
                       header=True,
                       float_format='%.2f')
        self.log.info('  Results: {}'.format(results_name))

    def load_model(self, model_basename):
        """
        load json and create model
        :param model_basename: model basename without extension. 'h5' and
        'json' will be added
        :return: the loaded model
        """
        json_file = open('{}.json'.format(model_basename), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.log.info("Loaded model from disk: {}.json".format(model_basename))

        # load weights into new model
        self.model.load_weights('{}.h5'.format(model_basename))
        self.log.info("Loaded weights from disk: {}.h5".format(model_basename))
        return self.model
