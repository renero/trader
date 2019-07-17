import os

from keras.layers import Dense, InputLayer
from keras.models import Sequential, model_from_json


class NN(object):

    def __init__(self, context_dictionary):
        self.__dict__.update(context_dictionary)

    def create_model(self) -> Sequential:
        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1, self._num_states)))
        model.add(
            Dense(self._num_states * self._num_actions,
                  input_shape=(self._num_states, ),
                  activation='sigmoid'))
        model.add(Dense(self._num_actions,
                        input_shape=(self._num_states * self._num_actions, ),
                        activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        model.summary()

        return model

    def save_model(self, model):
        # Check if file exists to abort saving operation
        solved = False
        char_to_append = ''
        while not solved:
            basename = 'model' + char_to_append + '.json'
            fname = os.path.join(self._models_dir, basename)
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
        print('Saved model to disk: {}'.format(fname))

        # serialize weights to HDF5
        basename = 'model' + char_to_append + '.h5'
        fname = os.path.join(self._models_dir, basename)
        model.save_weights(fname)
        print('Saved weights to disk: {}'.format(fname))

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
