import pickle
from os.path import basename, join, splitext, dirname, realpath
from pathlib import Path

import numpy as np
import pandas as pd

from oh_encoder import OHEncoder
from utils.file_io import file_exists, valid_output_name
from utils.strings import which_string, previous


class CSEncoder:
    """Takes as init argument a numpy array with 4 values corresponding
    to the O, H, L, C values, in the order specified by the second argument
    string `encoding` (for instance: 'ohlc').
    """

    cse_zero_open = 0.0
    cse_zero_high = 0.0
    cse_zero_low = 0.0
    cse_zero_close = 0.0
    fitted = False

    onehot = {}

    min_relative_size = 0.02
    shadow_symmetry_diff_threshold = 0.1
    movement_columns = ['open', 'high', 'low', 'close']
    diff_tags = ['open', 'close', 'high', 'low', 'min', 'max']
    def_enc_body_groups = ['ABCDE', 'FGHIJ', 'KLMNO', 'PQRST', 'UVWXY', 'Z']
    def_enc_body_sizes = [0.0, 0.10, 0.250, 0.50, 0.75, 1.0]
    cs_shift = [0.0, +1.0, -1.0, +2.0, -2.0]
    def_mvmt_upper_limits = [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ]
    def_mvmt_thresholds = [
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02
    ]
    def_prcntg_body_encodings = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]
    def_prcntg_mvmt_encodings = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'
    ]

    def __init__(self, params, values=None, encoding="ohlc"):
        """
        Takes as init argument a numpy array with 4 values corresponding
        to the O, H, L, C values, in the order specified by the second argument
        string `encoding` (for instance: 'ohlc').
        """
        self.params = params
        self.log = params.log

        self.encoding = encoding.upper()
        self.open = 0.0
        self.close = 0.0
        self.high = 0.0
        self.low = 0.0
        self.min = 0.0
        self.max = 0.0
        self.min_percentile = 0.0
        self.max_percentile = 0.0
        self.mid_body_percentile = 0.0
        self.mid_body_point = 0.0
        self.positive = False
        self.negative = False
        self.has_upper_shadow = False
        self.has_lower_shadow = False
        self.has_both_shadows = False
        self.shadows_symmetric = False
        self.body_in_upper_half = False
        self.body_in_lower_half = False
        self.body_in_center = False
        self.hl_interval_width = 0.0
        self.upper_shadow_len = 0.0
        self.lower_shadow_len = 0.0
        self.upper_shadow_percentile = 0.0
        self.lower_shadow_percentile = 0.0
        self.oc_interval_width = 0.0
        self.body_relative_size = 0.0
        self.shadows_relative_diff = 0.0

        # Save the origin of data here.
        self.dataset = splitext(basename(self.params.input_file))[0]

        # Assign the proper values to them
        err = 'Could not find all mandatory chars (o, h, l, c) in encoding ({})'
        if values is not None:
            if self.correct_encoding() is False:
                raise ValueError(err.format(self.encoding))
            self.open = values[self.encoding.find('O')]
            self.high = values[self.encoding.find('H')]
            self.low = values[self.encoding.find('L')]
            self.close = values[self.encoding.find('C')]
            self.calc_parameters()

        # Assign default encodings for movement
        self.encoded_delta_close = 'pA'
        self.encoded_delta_high = 'pA'
        self.encoded_delta_low = 'pA'
        self.encoded_delta_max = 'pA'
        self.encoded_delta_min = 'pA'
        self.encoded_delta_open = 'pA'

    def body_dict(self):
        return np.array(self.def_prcntg_body_encodings)

    def move_dict(self):
        return np.array(self.def_prcntg_mvmt_encodings)

    @classmethod
    def build_new(cls, params, values):
        return cls(params, values)

    def fit(self, ticks):
        self.log.info('Fitting CS encoder to ticks read.')
        self.cse_zero_open = ticks.loc[ticks.index[0], 'o']
        self.cse_zero_high = ticks.loc[ticks.index[0], 'h']
        self.cse_zero_low = ticks.loc[ticks.index[0], 'l']
        self.cse_zero_close = ticks.loc[ticks.index[0], 'c']
        self.fitted = True
        self.add_ohencoder()
        return self

    def add_ohencoder(self):
        # Create the OneHot encoders associated to each part of the data
        # which are the moment are 'body' and 'move'. Those names are extracted
        # from the parameters file.
        self.log.info(
            'Adding OneHot encoders for names {}'.format(self.params.subtypes))
        for name in self.params.subtypes:
            call_dict = getattr(self, '{}_dict'.format(name))
            self.onehot[name] = OHEncoder(self.params).fit(call_dict())

    @staticmethod
    def div(a, b):
        b = b + 0.000001 if b == 0 else b
        return a / b

    def calc_parameters(self):
        # positive or negative movement
        if self.close > self.open:
            self.max = self.close
            self.min = self.open
            self.positive = True
        else:
            self.max = self.open
            self.min = self.close
            self.negative = True

        # Length of the interval between High and Low
        self.hl_interval_width = abs(self.high - self.low)
        self.oc_interval_width = self.max - self.min

        # Mid point of the body (absolute value)
        self.mid_body_point = self.min + (self.oc_interval_width / 2.0)
        # Percentile of the body (relative)
        self.mid_body_percentile = self.div((self.mid_body_point - self.low),
                                            self.hl_interval_width)

        # Calc the percentile position of min and max values
        self.min_percentile = self.div((self.min - self.low),
                                       self.hl_interval_width)
        self.max_percentile = self.div((self.max - self.low),
                                       self.hl_interval_width)

        # total candle interval range width and shadows lengths
        self.upper_shadow_len = self.high - self.max
        self.upper_shadow_percentile = self.div(self.upper_shadow_len,
                                                self.hl_interval_width)
        self.lower_shadow_len = self.min - self.low
        self.lower_shadow_percentile = self.div(self.lower_shadow_len,
                                                self.hl_interval_width)

        # Percentage of HL range occupied by the body.
        self.body_relative_size = self.div(self.oc_interval_width,
                                           self.hl_interval_width)

        # Upper and lower shadows are larger than 2% of the interval range len?
        if self.div(self.upper_shadow_len,
                    self.hl_interval_width) > self.min_relative_size:
            self.has_upper_shadow = True
        if self.div(self.lower_shadow_len,
                    self.hl_interval_width) > self.min_relative_size:
            self.has_lower_shadow = True
        if self.has_upper_shadow and self.has_lower_shadow:
            self.has_both_shadows = True

        # Determine if body is centered in the interval. It must has
        # two shadows with a difference in their lengths less than 5% (param).
        self.shadows_relative_diff = abs(self.upper_shadow_percentile -
                                         self.lower_shadow_percentile)
        if self.has_both_shadows is True:
            if self.shadows_relative_diff < self.shadow_symmetry_diff_threshold:
                self.shadows_symmetric = True

        # Is body centered, or in the upper or lower half?
        if self.min_percentile > 0.5:
            self.body_in_upper_half = True
        if self.max_percentile < 0.5:
            self.body_in_lower_half = True
        if self.shadows_symmetric is True and self.body_relative_size > self.min_relative_size:
            self.body_in_center = True
        # None of the above is fulfilled...
        if any([
            self.body_in_center, self.body_in_lower_half,
            self.body_in_upper_half
        ]) is False:
            if self.lower_shadow_percentile > self.upper_shadow_percentile:
                self.body_in_upper_half = True
            else:
                self.body_in_lower_half = True

    def correct_encoding(self):
        """
        Check if the encoding proposed has all elements (OHLC)
        :return: True or False
        """
        # check that we have all letters
        return all(self.encoding.find(c) != -1 for c in 'OHLC')

    def encode_with(self, encoding_substring):
        err_msg = 'Body centered & 2 shadows but not in upper or lower halves'
        if self.body_in_center:
            # print('  centered')
            return encoding_substring[0]
        else:
            if self.has_both_shadows:
                if self.body_in_upper_half:
                    return encoding_substring[1]
                else:
                    if self.body_in_lower_half:
                        return encoding_substring[2]
                    else:
                        raise ValueError(err_msg)
            else:
                if self.has_lower_shadow:
                    return encoding_substring[3]
                else:
                    return encoding_substring[4]

    def __encode_body(self):
        if self.body_relative_size <= self.min_relative_size:
            return self.encode_with('ABCDE')
        else:
            if self.body_relative_size <= 0.1 + 0.05:
                # print('  10%')
                return self.encode_with('FGHIJ')
            else:
                if self.body_relative_size <= 0.25 + 0.1:
                    # print('  25%')
                    return self.encode_with('KLMNO')
                else:
                    if self.body_relative_size <= 0.5 + 0.1:
                        # print('  50%')
                        return self.encode_with('PQRST')
                    else:
                        if self.body_relative_size <= 0.75 + 0.1:
                            # print('  75%')
                            return self.encode_with('UVWXY')
                        else:
                            # print('  ~ 100%')
                            return 'Z'

    def encode_body(self):
        if self.positive:
            first_letter = 'p'
        else:
            first_letter = 'n'
        encoding = first_letter + self.__encode_body()
        setattr(self, 'encoded_body', encoding)
        # return encoding

    def encode_body_nosign(self):
        encoding = self.__encode_body()
        setattr(self, 'encoded_body', encoding)
        return encoding

    def __encode_movement(self,
                          value,
                          encoding=None,
                          upper_limits=def_mvmt_upper_limits,
                          thresholds=def_mvmt_thresholds,
                          encodings=def_prcntg_mvmt_encodings,
                          pos=0):
        """Tail recursive function to encode a value in one of the possible
        encodings passed as list. The criteria is whether the value is lower
        than a given upper_limit + threshold to use that encoding position.
        If not, jump to the next option until we find the upper limit, or
        move beyond limits.

        Args:
            value(float)  : the value to be encoded
            encoding (str): the resulting encoding found. Must be None in the
                            first call to the function.
            upper_limits(arr[float]): list of the upper limits to consider in
                            the coding schema to be used.
            thresholds(arr[float]): list of threshold to be added to the upper
                            limits.
            encodings (arr[str]): the list of resulting encodings, one for
                            each upper limit, plus one for the beyond last
                            limit case.
            pos:            controls recursion, must be 0 in the first call

        """
        if encoding is not None:
            return encoding
        if pos == len(encodings) - 1:
            # print('>> beyond limimts')
            return encodings[pos]
        if value <= upper_limits[pos] + thresholds[pos]:
            encoding = encodings[pos]
        return self.__encode_movement(value, encoding, upper_limits,
                                      thresholds, encodings, pos + 1)

    def encode_movement(self, prev_cs):
        """Compute the percentage of change for the OHLC values with respect
        to the relative range of the previous candlestick object (passed as
        argument). This allows to encode the movement of single candlestick.
        """
        for attr in self.diff_tags:
            delta = self.div((getattr(self, attr) - getattr(prev_cs, attr)),
                             prev_cs.hl_interval_width)
            encoding = self.__encode_movement(delta)
            if delta >= 0.0:
                sign_letter = 'p'
            else:
                sign_letter = 'n'
            self.log.debug(
                'Enc. {}; Delta={:.2f} ({:.2f} -> {:.2f}) as <{}>'.format(
                    attr, delta, getattr(prev_cs, attr), getattr(self, attr),
                    encoding))
            setattr(self, 'delta_{}'.format(attr), delta)
            setattr(self, 'encoded_delta_{}'.format(attr), '{}{}'.format(
                sign_letter, encoding))

    def adjust_body(self, letter, tick):
        """Given an encoding letter used for the body of the CS, return the
        expected positions of the upper and lower parts of the body, according
        to the encoding rules.
        Parameters:
          - letter: the letter of the body encoding that determines the size
                    of the CS as a percentage of the total height of the candle
          - tick: the tick that needs to be adjusted with the body encoding
                  information.
        """
        # the letter is the second character in the string.
        self.log.debug(
            '>> Adjusting tick: {:.02f}|{:.02f}|{:.02f}|{:.02f}'.format(
                tick[0], tick[1], tick[2], tick[3]))
        (block, pos) = which_string(self.def_enc_body_groups, letter)
        body_size = self.def_enc_body_sizes[block]
        self.log.debug('   letter ({}) => body size: {:.2f}'.format(
            letter, body_size))
        # High - Low is the height range the adjustment refers to.
        tick_range = tick[1] - tick[2]
        M = 0.5 + (body_size / 2.0)
        m = 0.5 - (body_size / 2.0)
        self.log.debug('   tick range: {:.2f}, M: {:.2f}, m: {:.2f}'.format(
            tick_range, M, m))
        shift = ((1.0 - M) / 2.0) * self.cs_shift[pos]
        self.log.debug('   shift = {:.2f}'.format(shift))
        if tick[0] < tick[3]:
            tick[0] = tick[2] + (m + shift) * tick_range
            tick[3] = tick[2] + (M + shift) * tick_range
        else:
            tick[3] = tick[2] + (m + shift) * tick_range
            tick[0] = tick[2] + (M + shift) * tick_range
        self.log.debug('<< New tick: {:.02f}|{:.02f}|{:.02f}|{:.02f}'.format(
            tick[0], tick[1], tick[2], tick[3]))
        return tick

    def decode_movement_code(self, code):
        sign = code[0]
        letter = code[1]
        pos = self.def_prcntg_mvmt_encodings.index(letter)
        self.log.debug('Percentage position <{}={}>'.format(letter, pos))
        value = self.def_mvmt_upper_limits[pos] if pos < len(
            self.def_mvmt_upper_limits) else self.def_mvmt_upper_limits[len(
            self.def_mvmt_upper_limits)]
        self.log.debug('New value = {:.2f}'.format(value))
        if sign == 'n':
            value *= -1.0
        self.log.debug('Decoding <{}> with value: {:.2f}'.format(code, value))
        return value

    def decode_cse(self, this_cse, prev_cse, col_names):
        """
        From a CSE numpy array and its previous CSE numpy array in the
        time series, returns the reconstructed tick (OHLC).
        """
        mm = prev_cse.hl_interval_width
        # Set the columns names that contain the values in my date
        amount_shift = [(self.decode_movement_code(this_cse[column]) * mm)
                        for column in col_names]
        self.log.debug(
            'Amount of movement: {:.04f}|{:.04f}|{:.04f}|{:.04f}'.format(
                amount_shift[0], amount_shift[1], amount_shift[2],
                amount_shift[3]))
        reconstructed_tick = [
            prev_cse.min + amount_shift[0],
            prev_cse.high + amount_shift[0],
            prev_cse.low + amount_shift[0],
            prev_cse.max + amount_shift[0]
        ]
        # If this CSE is negative, swap the open and close values
        if this_cse['b'][0] == 'n':
            self.log.debug('This CS seems to be negative')
            open = reconstructed_tick[3]
            close = reconstructed_tick[0]
            reconstructed_tick[0] = open
            reconstructed_tick[3] = close
        self.log.debug(
            '>> Reconstructed tick: {:.02f}|{:.02f}|{:.02f}|{:.02f}'.format(
                reconstructed_tick[0], reconstructed_tick[1],
                reconstructed_tick[2], reconstructed_tick[3]))
        return reconstructed_tick

    # TODO: Clarify when 'd' is present in
    def cse2ticks(self, cse_codes, first_cse, col_names=None):
        """Reconstruct CSE codes read from a CSE file into ticks
        Arguments
          - cse_codes: DataFrame with columns 'b', 'o', 'h', 'l', 'c',
            representing the body of the candlestick, the open, high, low and
            close encoded values as two-letter strings.
        Returns:
          - A DataFrame with the open, high, low and close values decoded.
          :param cse_codes: the list of CSEs to convert back to Ticks
          :param first_cse: the first CSE to use as reference
          :param col_names: the names of column headers to use with ticks
          :return: the ticks as a dataframe.
        """
        assert self.fitted, "The encoder has not been fit with data yet!"
        if col_names is None:
            col_names = list(self.params.csv_dict.keys())
            # Remove date column if present.
            if 'd' in col_names:
                col_names.remove('d')
        cse_decoded = [first_cse]
        self.log.debug('Zero CS created: {:.2f}|{:.2f}|{:.2f}|{:.2f}'.format(
            self.cse_zero_open, self.cse_zero_high, self.cse_zero_low,
            self.cse_zero_close))
        rec_ticks = [[
            first_cse.open, first_cse.high, first_cse.low, first_cse.close
        ]]
        for i in range(0, len(cse_codes)):
            this_cse = cse_codes.loc[cse_codes.index[i]]
            self.log.debug('Decoding: {}|{}|{}|{}|{}'.format(
                this_cse['b'], this_cse['o'], this_cse['h'], this_cse['l'],
                this_cse['c']))
            this_tick = self.decode_cse(this_cse, cse_decoded[-1], col_names)
            cse_decoded.append(self.build_new(self.params, this_tick))
            this_tick = self.adjust_body(this_cse['b'][1], this_tick)
            self.log.debug(
                'Adjusted CS body: {:.2f}|{:.2f}|{:.2f}|{:.2f}'.format(
                    this_tick[0], this_tick[1], this_tick[2], this_tick[3]))
            rec_ticks.append(this_tick)

        result = pd.DataFrame(rec_ticks)
        result.columns = col_names
        return result

    def encode_tick(self, tick, prev_cse):
        cse = CSEncoder(self.params, np.array(tick))
        self.log.debug(
            'Tick encoding: [{:.2f}|{:.2f}|{:.2f}|{:.2f}]'.format(
                cse.open, cse.high, cse.low, cse.close))
        cse.encode_body()
        if prev_cse is None:
            cse.encode_movement(cse)
        else:
            cse.encode_movement(prev_cse)
        return cse

    def ticks2cse(self, ticks):
        """
        Encodes a dataframe of Ticks, returning an array of CSE objects.
        """
        self.log.debug('Converting ticks dim{} to CSE.'.format(ticks.shape))
        cse = []
        for index in range(0, ticks.shape[0]):
            cse.append(
                self.encode_tick(ticks.iloc[index], previous(cse, index)))
        return cse

    def read_cse(self, filename=None, col_names=None):
        if filename is None:
            df = pd.read_csv(self.params.cse_file, sep=',')
        else:
            df = pd.read_csv(filename, sep=',')
        df.columns = col_names if col_names is not None else self.params.cse_colnames
        return df

    def save(self):
        """
        Saves the CS Encoder object into a pickle dump.
        :return: The objects itself
        """
        encoder_filename = valid_output_name(
            'encoder_{}_w{}'.format(self.dataset, self.params.window_size),
            path=self.params.predictions_path,
            extension='pickle'
        )
        with open(encoder_filename, 'wb') as f:
            # Pickle the object dictionary using the highest protocol available
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            self.log.info('Saved encoder to: {}'.format(encoder_filename))
        return self

    def load(self, pickle_file_path=None):
        """
        Loads the CS Encoder object from a pickle dump.
        :return: The objects itself
        """
        # Keep current parameters
        current_log_level = self.params.log_level
        current_input_file = self.params.input_file

        if pickle_file_path is None:
            path = self.params.pickle_filename
        else:
            path = pickle_file_path
        pickle_file = join(self.params.models_dir, '{}.pickle'.format(path))
        pickle_file = file_exists(pickle_file, dirname(realpath(__file__)))
        with open(pickle_file, 'rb') as f:
            self.__dict__.update(pickle.load(f).__dict__)

        # Overwrite some params from pickle file.
        self.log.set_level(current_log_level)
        self.params.input_file = current_input_file

        self.log.info('Loaded encoder pickle file: {}'.format(pickle_file))
        self.add_ohencoder()

        return self

    def save_cse(self, cse, filename):
        """
        Saves a list of CSE objects to the filename specifed.
        Arguments:
            - cse(list(CSEEncoder)): list of CSE objects
            - filename: the path to the file to be written as CSV
        """
        my_file = Path(filename)
        if my_file.is_file():
            self.log.warn('No writing to file as file alrady exists')
            return

        body = [cse[i].encoded_body for i in range(len(cse))]
        delta_open = [cse[i].encoded_delta_min for i in range(len(cse))]
        delta_high = [cse[i].encoded_delta_high for i in range(len(cse))]
        delta_low = [cse[i].encoded_delta_low for i in range(len(cse))]
        delta_close = [cse[i].encoded_delta_max for i in range(len(cse))]

        df = pd.DataFrame(
            data={
                'body': body,
                'open': delta_open,
                'high': delta_high,
                'low': delta_low,
                'close': delta_close
            })
        df.to_csv(filename, sep=',', index=False)

    def encode(self, ticks):
        """
        Encodes a dataframe of Ticks, returning a dataframe of CSE values.
        """
        cse = []
        df = pd.DataFrame(index=range(ticks.shape[0]),
                          columns=self.params.cse_tags)
        for index in range(0, ticks.shape[0]):
            cse.append(
                self.encode_tick(ticks.iloc[index], previous(cse, index)))
            df.loc[index] = pd.Series({
                self.params.cse_tags[0]: cse[index].encoded_body,
                self.params.cse_tags[1]: cse[index].encoded_delta_open,
                self.params.cse_tags[2]: cse[index].encoded_delta_high,
                self.params.cse_tags[3]: cse[index].encoded_delta_low,
                self.params.cse_tags[4]: cse[index].encoded_delta_close})
        return df

    def info(self):
        v = vars(self)
        for key, value in sorted(v.items(), key=lambda x: x[0]):
            if isinstance(value, np.float64):
                print('{:.<25}: {:>.3f}'.format(key, value))
            elif isinstance(value, str) or isinstance(value, int):
                print('{:.<25}: {:>}'.format(key, value))

    def values(self):
        print('O({:.3f}), H({:.3f}), L({:.3f}), C({:.3f})'.format(
            self.open, self.high, self.low, self.close))

    @classmethod
    def body(self, cse):
        """Returns the body element of an array of encoded candlesticks"""
        bodies = np.array([cse[i].encoded_body for i in range(len(cse))])
        return pd.DataFrame(bodies, columns=['body'])

    @classmethod
    def move(self, cse):
        """Returns the body element of an array of encoded candlesticks"""
        ohlc = np.array([[
            cse[i].encoded_delta_open, cse[i].encoded_delta_high,
            cse[i].encoded_delta_low, cse[i].encoded_delta_close
        ] for i in range(len(cse))])
        return pd.DataFrame(ohlc, columns=self.movement_columns)
