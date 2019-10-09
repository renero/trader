import os
from os.path import splitext, basename

from keras.layers import Dense, InputLayer
from keras.models import Sequential, model_from_json

from common import Common
from file_io import valid_output_name
from utils.dictionary import Dictionary


class RL_NN(Common):

    def __init__(self, configuration: Dictionary):
        self.params = configuration

    def create_model(self) -> Sequential:
        num_cells = int(
            self.params.num_states * \
            self.params.num_actions * \
            self.params.cells_reduction_factor)
        self.log('Default cells: {}, but truncated to 64'.format(num_cells))
        num_cells = 64

        model = Sequential()
        model.add(
            InputLayer(batch_input_shape=(None, self.params.num_states)))
        model.add(Dense(
            num_cells,
            input_shape=(self.params.num_states,),
            activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(
            Dense(
                self.params.num_actions,
                input_shape=(
                    num_cells,),
                activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        if self.params.debug is True:
            self.log('Model Summary')
            model.summary()

        return model

    def save_model(self, model):
        self.log('\nSaving model, weights and results.')

        fname = 'rl_model_' + splitext(
            basename(self.params.data_path))[0]
        model_name = valid_output_name(fname, self.params.models_dir, 'json')

        # serialize model to JSON
        model_json = model.to_json()
        with open(model_name, 'w') as json_file:
            json_file.write(model_json)
        self.log('  Model: {}'.format(model_name))

        # Serialize weights to HDF5
        weights_name = model_name.replace('.json', '.h5')
        model.save_weights(weights_name)
        print('  Weights: {}'.format(weights_name))

        # Save also the results table
        results_name = model_name.replace('.json', '.csv')
        self.params.results.to_csv(results_name,
                                   sep=',',
                                   index=False,
                                   header=True,
                                   float_format='%.2f')
        print('  Results: {}'.format(results_name))

    def load_model(self, model, weights):
        """
        load json and create model
        :param model:
        :param weights:
        :return:
        """
        json_file = open(model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        print("Loaded model from disk: {}".format(model))

        # load weights into new model
        loaded_model.load_weights(weights)
        print("Loaded model from disk: {}".format(weights))
        return loaded_model
