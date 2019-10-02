from os.path import splitext, basename

import pandas as pd
from pandas import DataFrame
from tabulate import tabulate

from cs_api import single_prediction
from cs_encoder import CSEncoder
from cs_nn import Csnn
from dataset import Dataset
from file_io import valid_output_name
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
        dataset = self.prepare_input(encoder, cse, self.subtypes)
        nn = self.train_nn(dataset, self.subtypes)
        encoder.save()

        return nn, encoder

    @staticmethod
    def train_nn(dataset, subtypes):
        """
        Train a model.
        """
        nn = {}
        for subtype in subtypes:
            nn[subtype] = Csnn(None, subtype)

            window_size = dataset[subtype].X_train.shape[1]
            num_categories = dataset[subtype].X_train.shape[2]

            nn[subtype].build_model(window_size, num_categories).train(
                dataset[subtype].X_train, dataset[subtype].y_train).save()

        return nn

    @staticmethod
    def load_nn(model_names, subtypes):
        """
        """
        nn = {}
        for name in model_names.keys():
            nn[name] = {}
            for subtype in subtypes:
                nn[name][subtype] = Csnn(name, subtype)
                nn[name][subtype].load(model_names[name][subtype])
        return nn

    @staticmethod
    def load_encoders(model_names):
        """Load a encoder for each network"""
        encoder = {}
        for name in model_names:
            encoder[name] = CSEncoder().load(model_names[name]['encoder'])
        return encoder

    def prepare_predict(self):
        nn = self.load_nn(self._model_names, self.subtypes)
        encoder = self.load_encoders(self._model_names)

        return nn, encoder

    def predict_training(self, data, nn, encoder, ticks) -> DataFrame:
        self.log.info('Performing prediction over TRAINING set')
        predictions = pd.DataFrame([])

        num_ticks = data.shape[0]
        max_wsize = max(
            [encoder[name]._window_size for name in self.model_names])
        train_range = range(0+max_wsize, num_ticks - 1)

        self.log.info('Predicting over {} training ticks'.format(num_ticks))
        self.log.info('Looping {} groups of {} ticks'.format(
            len(train_range), max_wsize))

        for from_idx in train_range:
            prediction = single_prediction(data, from_idx, nn, encoder, self)
            prediction = self.add_supervised_info(
                prediction,
                data.iloc[from_idx]['c'], self)
            predictions = predictions.append(prediction)
        predictions = ticks.scale_back(predictions)

        return predictions

    def predict_newdata(self, data, nn, encoder, ticks) -> DataFrame:
        self.log.info('Performing prediction over NEW UNKNOWN data')
        predictions = pd.DataFrame([])

        # Take only the last window_size+1 elements, leaving
        # the last as the actual value
        prediction = single_prediction(data, -1, nn, encoder, self)
        predictions = ticks.scale_back(predictions.append(prediction))

        return predictions

    @staticmethod
    def add_supervised_info(prediction, real_value, params):
        """
        Add to the single prediction the actual value expected (if any), the
        difference between the mean predicted value and the actual value, and
        which network name is the one producing the closest value

        :param prediction: the prediction made by all networks in a form of
        DataFrame with the columns being the name of the networks
        :param real_value: the actual value that followed the sequence presented
        to the set of networks. This is the value to be predicted.
        :param params: The parameters of the whole enchilada

        :return: The dataframe of the predictions enriched with the actual
        prediction, the difference between the mean and the actual, and the winner
        network.
        """

        def diff_with(column_label):
            return abs(prediction['actual'] - prediction[column_label])

        model_names = list(params.model_names.keys())
        prediction['actual'] = real_value
        if len(model_names) > 1:
            prediction['avg_diff'] = diff_with('avg')
            prediction['med_diff'] = diff_with('median')
            if params._ensemble is True:
                prediction['ens_diff'] = diff_with('ensemble')
            max_diff = 10000000.0
            winner = ''
            for name in params.model_names.keys():
                diff = abs(
                    prediction[name].iloc[-1] - prediction['actual'].iloc[-1])
                if diff < max_diff:
                    max_diff = diff
                    winner = name
            prediction.loc[:, 'winner'] = winner
        return prediction

    @staticmethod
    def reorder_predictions(predictions, params):
        """
        Reorder predictions: If I'm producing predictions for the training set I
        reorder 'actual' values column in first column position. If I'm only
        producing a single prediction then, I simply drop all columns referring
        to the case when I know the response.
        :param predictions: the data frame with the predictions
        :param params: the parameters file.
        :return: the predictions reordered.
        """
        if params.num_models == 1 and params._predict_training is False:
            return predictions

        if params._predict_training is False:
            cols_to_drop = ['actual', 'avg', 'avg_diff', 'med_diff', 'median']
            if 'winner' in predictions.columns:
                cols_to_drop += ['winner']
            predictions = predictions.drop(cols_to_drop, axis=1)
        else:
            # Reorder columns to set 'actual' in first position
            actual_position = list(predictions.columns).index('actual')
            num_cols = len(predictions.columns)
            columns = [actual_position] + \
                      [i for i in range(actual_position)] + \
                      [j for j in range(actual_position + 1, num_cols)]
            predictions = predictions.iloc[:, columns]

        return predictions

    @staticmethod
    def save_predictions(predictions, params, log):
        if params._save_predictions is not True:
            log.info('not saving predictions.')
            return
        # Find a valid filename and save everything
        filename = valid_output_name(
            filename='pred_{}_{}'.format(
                splitext(basename(params._ticks_file))[0],
                '_'.join(params.model_names)),
            path=params._predictions_path,
            extension='csv')
        predictions.to_csv(filename, index=False)
        log.info('predictions saved to: {}'.format(filename))

    def display_predictions(self, predictions):
        print(tabulate(predictions, headers='keys',
                       tablefmt=self._table_format,
                       showindex=False,
                       floatfmt=['.1f']))

    @staticmethod
    def prepare_input(encoder, cse, subtypes):
        """
        Prepare the training and test datasets from an list of existing CSE, for
        each of the model names considered (body and move).

        :param encoder: The encoder used to build the CSE list.
        :param cse: The list of CSE objects
        :param subtypes: The parameters read from file
        :return: The datasets for each of the models that need to be built. The
            names of the models specify the 'body' part and the 'move' part.
        """
        cse_data = {}
        oh_data = {}
        dataset = {}
        for subtype in subtypes:
            call_select = getattr(encoder, '{}'.format(subtype))
            cse_data[subtype] = Dataset().adjust(call_select(cse))
            oh_data[subtype] = encoder.onehot[subtype].encode(cse_data[subtype])
            dataset[subtype] = Dataset().train_test_split(oh_data[subtype])

        return dataset
