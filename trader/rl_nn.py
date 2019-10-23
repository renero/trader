from os.path import splitext, basename

from keras.layers import Dense, InputLayer
from keras.models import Sequential, model_from_json

from common import Common
from file_io import valid_output_name
from utils.dictionary import Dictionary


class RL_NN(Common):

    def __init__(self, configuration: Dictionary):
        self.params = configuration
        self.log = self.params.log

    def create_model(self) -> Sequential:
        model = Sequential()

        # Input layer
        model.add(
            InputLayer(batch_input_shape=(None, self.params.num_states)))
        first_layer = True

        # Create all the layers
        for num_cells in self.params.deep_qnet.hidden_layers:
            if first_layer:
                model.add(Dense(
                    num_cells,
                    input_shape=(self.params.num_states,),
                    activation=self.params.deep_qnet.activation))
                first_layer = False
            else:
                model.add(Dense(num_cells,
                                activation=self.params.deep_qnet.activation))

        # Output Layer
        last_layer_cells = self.params.deep_qnet.hidden_layers[-1]
        model.add(
            Dense(self.params.num_actions, input_shape=(last_layer_cells,),
                  activation='linear'))
        model = self.compile_model(model)
        if self.params.debug and self.params.log_level > 2:
            self.log.debug('Model Summary')
            model.summary()

        return model

    def compile_model(self, model):
        model.compile(
            loss=self.params.deep_qnet.loss,
            optimizer=self.params.deep_qnet.optimizer,
            metrics=self.params.deep_qnet.metrics)
        return model

    def save_model(self, model, results):
        self.log.info('\nSaving model, weights and results.')

        fname = 'rl_model_' + splitext(
            basename(self.params.data_path))[0]
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
        loaded_model = model_from_json(loaded_model_json)
        self.log.info("Loaded model from disk: {}.json".format(model_basename))

        # load weights into new model
        loaded_model.load_weights('{}.h5'.format(model_basename))
        self.log.info("Loaded weights from disk: {}.h5".format(model_basename))
        return loaded_model
