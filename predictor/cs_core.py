import pandas as pd

from cs_api import split_datasets, train_nn, single_prediction, \
    add_supervised_info, load_nn, load_encoders
from cs_encoder import CSEncoder
from params import Params


class CSCore(Params):

    def __init__(self):
        super(CSCore, self).__init__()

    def train(self, data):
        """
        Train networks to the data (OHLC) passed
        :param data: Data in OHLC format from the ticks module.
        :return: the NN trained, and the encoder used
        """
        encoder = CSEncoder().fit(data)
        cse = encoder.ticks2cse(data)
        dataset = split_datasets(encoder, cse, self.subtypes)
        nn = train_nn(dataset, self.subtypes)
        encoder.save()

        return nn, encoder

    def prepare_predict(self):
        nn = load_nn(self._model_names, self.subtypes)
        encoder = load_encoders(self._model_names)

        return nn, encoder

    def predict_training(self, data, nn, encoder, ticks):
        predictions = pd.DataFrame([])

        # TODO: I'm loading the first encoder, but I do need to use
        #       all of them, in case I'm loading more than one network
        cse = encoder[next(iter(self._model_names))].ticks2cse(data)

        # TODO: I'm producing only 25 predictions here, to speed up testing.
        num_ticks = data.shape[0]
        self.log.info('num ticks: {}'.format(num_ticks))
        # train_range = range(0, ticks.shape[0] - params._window_size + 1)
        train_range = range(num_ticks - 25, num_ticks - self._window_size)
        self.log.info('range: {}'.format(train_range))

        for from_idx in train_range:
            self.log.info('processing [{}:{}]'.format(from_idx, from_idx+self._window_size))
            tick_group = data.iloc[from_idx:from_idx + self._window_size]
            prediction = single_prediction(tick_group, nn, encoder, self)
            prediction = add_supervised_info(
                prediction,
                data.iloc[from_idx + self._window_size]['c'],
                self)
            predictions = predictions.append(prediction)
        predictions = ticks.scale_back(predictions)

        return predictions

    def predict_newdata(self, data, nn, encoder, ticks):
        predictions = pd.DataFrame([])

        # Take only the last window_size+1 elements, leaving
        # the last as the actual value
        tick_group = data.tail(self._window_size + 1).iloc[
                     -self._window_size - 1:-1]
        prediction = single_prediction(tick_group, nn, encoder, self)
        # Take the last element as the supervised information.
        prediction = add_supervised_info(prediction, data.iloc[-1]['c'],
                                         self)
        predictions = ticks.scale_back(predictions.append(prediction))

        return predictions
