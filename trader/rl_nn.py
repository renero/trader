import os
from os.path import splitext, basename

from keras.layers import Dense, InputLayer
from keras.models import Sequential, model_from_json

from common import Common
from utils.dictionary import Dictionary


class RL_NN(Common):

    def __init__(self, configuration: Dictionary):
        self.configuration = configuration

    def create_model(self) -> Sequential:
        num_cells = int(
            self.configuration.num_states * \
            self.configuration.num_actions * \
            self.configuration.cells_reduction_factor)
        self.log('Default cells: {}, but truncated to 64'.format(num_cells))
        num_cells = 64

        model = Sequential()
        model.add(
            InputLayer(batch_input_shape=(1, self.configuration.num_states)))
        model.add(Dense(
            num_cells,
            input_shape=(self.configuration.num_states,),
            activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(
            Dense(
                self.configuration.num_actions,
                input_shape=(
                    num_cells,),
                activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        if self.configuration.debug is True:
            self.log('Model Summary')
            model.summary()

        return model

    def save_model(self, model):
        self.log('\nSaving model, weights and results.')
        # Check if file exists to abort saving operation
        solved = False
        char_to_append = ''
        fname = 'rl_model_' + splitext(basename(self.configuration.data_path))[0]
        while not solved:
            base_filename = fname + char_to_append + '.json'
            fname = os.path.join(self.configuration.models_dir, base_filename)
            if os.path.isfile(fname) is not True:
                solved = True
            else:
                if char_to_append == '':
                    char_to_append = str('1')
                else:
                    char_to_append = str(int(char_to_append) + 1)

        # serialize model to JSON
        model_json = model.to_json()
        with open(fname, 'w') as json_file:
            json_file.write(model_json)
        self.log('  Model: {}'.format(fname))

        # Serialize weights to HDF5
        base_filename = fname + char_to_append + '.h5'
        fname = os.path.join(self.configuration.models_dir, base_filename)
        model.save_weights(fname)
        print('  Weights: {}'.format(fname))

        # Save also the results table
        base_filename = fname + char_to_append + '.csv'
        fname = os.path.join(self.configuration.models_dir, base_filename)
        self.configuration.results.to_csv(fname,
                                          sep=',',
                                          header=True,
                                          float_format='%.2f')
        print('  Results: {}'.format(fname))

    def load_model(self, model, weights):
        # load json and create model
        json_file = open(model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        print("Loaded model from disk: {}".format(model))

        # load weights into new model
        loaded_model.load_weights(weights)
        print("Loaded model from disk: {}".format(weights))
        return loaded_model