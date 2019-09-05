import numpy as np
import pandas as pd
from params import Params
from keras.utils import to_categorical


class ValidationError(Exception):
    def __init__(self, message, errors):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


class OHEncoder(Params):
    _signed = False
    _states = set()
    _dict = dict()
    _inv_dict = dict()
    _sign_dict = {'p': +1, 'n': -1}
    _inv_sign = {1: 'p', -1: 'n'}

    def __init__(self, signed=True):
        super(OHEncoder, self).__init__()
        self._signed = signed

    def reset(self):
        self._states = set()
        self._dict = dict()
        self._inv_dict = dict()
        return self

    def fit(self, data):
        """Obtain the set of unique strings used in the data to shape
        a dictionary with a numeric mapping for each of them

        ## Arguments:
          - data: Either, a 1D or 2D array of strings. If the attribute
          'signed' is set to True when creating the encoder, the
          first character of every string is suposed to encode the
          sign of the element, so instead of encoding it as
          `[0, 0, ..., 1, 0 ... 0]`, it will be encoded as
          `0, 0, ..., -1, 0 ... 0]` with -1 if the sign is negative.

        ## Return Values:
          - The object, updated.
        """
        # Check if the array is 1D or 2D, and first element has more than 1 ch
        self.reset()
        if len(data.shape) == 2:
            if self._signed is True and len(data[0]) > 1:
                self.log.debug('case 1')
                [self._states.update([char[1:] for char in l]) for l in data]
            else:
                self.log.debug('case 2')
                [self._states.update(l) for l in data]
        elif len(data.shape) == 1:
            if self._signed is True and len(data[0]) > 1:
                self.log.debug('case 3')
                self._states.update([char[1:] for char in data])
            else:
                self.log.debug('case 4')
                self._states.update(data)
        else:
            raise ValidationError('1D or 2D array expected.', -1)
        # Build the dict.
        self.log.debug('Onehot encoding with {} elements'.format(
            len(self._states)))
        self._dict = {k: v for v, k in enumerate(sorted(list(self._states)))}
        self._inv_dict = {v: k for k, v in self._dict.items()}
        return self

    def fit_from_dict(self, data):
        """ DEPRECATED
        """
        if len(data.shape) == 1:
            self._states.update(data)
        else:
            raise ValidationError('1D array expected as dictionary.', -1)
        self._dict = {k: v for v, k in enumerate(sorted(list(self._states)))}
        self._inv_dict = {v: k for k, v in self._dict.items()}
        return self

    def encode(self, input_vector):
        """ Convert a DataFrame of dimension (n x m x p) into an array of
        ((nxm) x p), in one hot encoding, with sign.
        Arguments
          - input: DataFrame of dimension (n * m * p)
        Return values
          - DataFrame of dimension ((n*m) * p)
        """
        data = input_vector.values
        if len(data.shape) == 1 or len(data.shape) == 2:
            num_arrays = data.shape[0] if len(data.shape) == 2 else 1
            num_strings = data.shape[1] if len(
                data.shape) == 2 else data.shape[0]
            num_states = len(self._states)
            data = data.reshape([num_arrays, num_strings])
            if self._signed is True:
                transformed = np.empty([num_arrays, num_strings, num_states])
                for i in range(num_arrays):
                    for j in range(num_strings):
                        code = to_categorical(
                            self._dict[data[i][j][1:]],
                            num_classes=len(self._states))
                        sign = self._sign_dict[data[i][j][0].lower()]
                        transformed[i][j] = np.dot(code, sign)
            else:
                transformed = np.array([
                    to_categorical(
                        [self._dict[x] for x in y],
                        num_classes=len(self._states)) for y in data
                ])
        else:
            raise ValidationError('1D or 2D array expected.', -1)

        info_msg = 'Onehot encoded input {} -> {}'
        self.log.info(info_msg.format(input_vector.shape, transformed.shape))

        return pd.DataFrame(transformed.reshape(len(input_vector), -1))

    def decode(self, data):
        if len(data.shape) == 1 or len(data.shape) == 2:
            num_arrays = data.shape[0] if len(data.shape) == 2 else 1
            num_strings = data.shape[1] if len(
                data.shape) == 2 else data.shape[0]
            data = data.reshape([num_arrays, num_strings])
            # decode_len = 2 if self._signed else 1
            decoded = []
            for i in range(num_arrays):
                flags = np.isin(data[i], [1, -1])
                flag_index = np.where(flags)[0][0]
                invcode = self._inv_dict[flag_index]
                sign = self._inv_sign[data[i][flag_index]]
                if self._signed:
                    decoded.append('{}{}'.format(sign, invcode))
                else:
                    decoded.append(invcode)
        else:
            raise ValidationError('1D or 2D array expected.', -1)
        return np.array(decoded)
