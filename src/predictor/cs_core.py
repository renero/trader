from os.path import splitext, basename
from typing import Dict, List

import pandas as pd
from pandas import DataFrame

from cs_api import single_prediction
from cs_encoder import CSEncoder
from cs_nn import CS_NN
from dataset import Dataset
from ticks_reader import TicksReader
from utils.file_io import valid_output_name


class CSCore:

    def __init__(self, params):
        self.params = params
        self.log = params.log

    def train(self, data: DataFrame):
        """
        Train networks to the data (OHLC) passed
        :param data: Data in OHLC format from the ticks module.
        :return: the NN trained, and the encoder used
        """
        # Remove the "Date" Column
        ticks = data.copy(deep=True).drop([self.params.csv_dict['d']], axis=1)
        # Train
        encoder = CSEncoder(self.params).fit(ticks)
        cse = encoder.transform(ticks)
        dataset = self.prepare_input(encoder, cse, self.params.subtypes)
        nn = self.train_nn(dataset, self.params.subtypes)
        encoder.save()

        return nn, encoder

    def train_nn(self, dataset, subtypes):
        """
        Train a model.
        """
        nn = {}
        for subtype in subtypes:
            nn[subtype] = CS_NN(self.params, None, subtype)

            window_size = dataset[subtype].X_train.shape[1]
            num_categories = dataset[subtype].X_train.shape[2]

            nn[subtype].build_model(window_size, num_categories).train(
                dataset[subtype].X_train, dataset[subtype].y_train).save()

        return nn

    def load_nn(self,
                model_names: Dict[str, str],
                subtypes: List) -> Dict[str, Dict]:
        """
        """
        nn = {}
        for name in model_names.keys():
            nn[name] = {}
            for subtype in subtypes:
                nn[name][subtype] = CS_NN(self.params, name, subtype)
                nn[name][subtype].load(model_names[name][subtype])
        return nn

    def load_encoders(self, model_names: List[str]) -> Dict[str, CSEncoder]:
        """Load a encoder for each network"""
        encoder = {}
        for name in model_names:
            encoder[name] = CSEncoder(self.params).load(
                model_names[name]['encoder'])
        return encoder

    def prepare_predict(self) -> (Dict[str, Dict], ):
        nn = self.load_nn(self.params.model_names, self.params.subtypes)
        encoder = self.load_encoders(self.params.model_names)

        return nn, encoder

    def predict_training(self,
                         data,
                         nn,
                         encoder,
                         ticks_reader: TicksReader) -> DataFrame:
        self.log.info('Performing prediction over TRAINING set')
        predictions = pd.DataFrame([])
        date_column = self.params.csv_dict['d']

        num_ticks = data.shape[0]
        max_wsize = max(
            [encoder[name].params.window_size for name in
             self.params.model_names])
        train_range = range(0 + max_wsize, num_ticks - 1)

        self.log.info('Predicting over {} training ticks'.format(num_ticks))
        self.log.info('Looping {} groups of {} ticks'.format(
            len(train_range), max_wsize))

        for from_idx in train_range:
            prediction = single_prediction(data, from_idx, nn, encoder,
                                           self.params)
            prediction = self.add_supervised_info(
                prediction,
                data.iloc[from_idx]['c'], self.params)
            prediction.insert(loc=0,
                              column=date_column,
                              value=data.iloc[from_idx]['Date'])
            predictions = predictions.append(prediction)
        predictions = pd.concat((
            predictions[date_column],
            ticks_reader.scale_back(
                predictions.loc[:, predictions.columns != date_column])),
            axis=1)

        return predictions

    def predict_newdata(self, data, nn, encoder, ticks) -> DataFrame:
        self.log.info('Performing prediction over NEW UNKNOWN data')
        predictions = pd.DataFrame([])

        # Take only the last window_size+1 elements, leaving
        # the last as the actual value
        prediction = single_prediction(data, -1, nn, encoder, self.params)
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
        prediction, the difference between the mean and the actual, and the
        winner network.
        """

        def diff_with(column_label):
            return abs(prediction['actual'] - prediction[column_label])

        model_names = list(params.model_names.keys())
        prediction['actual'] = real_value
        if len(model_names) > 1:
            prediction['avg_diff'] = diff_with('avg')
            prediction['med_diff'] = diff_with('median')
            if params.ensemble is True:
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
        if params.num_models == 1 and params.predict_training is False:
            return predictions

        if params.predict_training is False:
            cols_to_drop = ['actual', 'avg', 'avg_diff', 'med_diff', 'median']
            if 'winner' in predictions.columns:
                cols_to_drop += ['winner']
            predictions = predictions.drop(cols_to_drop, axis=1)
        else:
            # Reorder columns to set 'actual' in first position
            date_column = params.csv_dict['d']
            actual_position = list(predictions.columns).index('actual')
            num_cols = len(predictions.columns)
            columns = [0, actual_position] + \
                      [i + 1 for i in range(actual_position - 1)] + \
                      [j for j in range(actual_position + 1, num_cols)]
            predictions = predictions.iloc[:, columns]

        return predictions

    @staticmethod
    def save_predictions(predictions, params, log):
        if params.save_predictions is not True:
            log.info('not saving predictions.')
            return
        # Find a valid filename and save everything
        if params.output is not None:
            name = params.output
        else:
            name = 'pred_{}_{}'.format(
                splitext(basename(params.input_file))[0],
                '_'.join(params.model_names))
        filename = valid_output_name(filename=name,
                                     path=params.predictions_path,
                                     extension='csv')
        predictions.round(2).to_csv(filename, index=False)
        log.info('predictions saved to: {}'.format(filename))

    def display_predictions(self, predictions):
        if self.params.predict:
            last_prediction = predictions.iloc[-1]
            print(pd.DataFrame(last_prediction).T.to_string(index=False))
            last_prediction.to_json(self.params.json_prediction)
            self.log.info('Save json file: {}'.format(
                self.params.json_prediction))
        else:
            pd.set_option('display.max_rows', -1)
            print(predictions.to_string(index=False))

    def prepare_input(self,
                      encoder: CSEncoder,
                      cse: List[CSEncoder],
                      subtypes: List[str]) -> Dict[str, Dataset]:
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
            cse_data[subtype] = Dataset(self.params).adjust(call_select(cse))
            oh_data[subtype] = encoder.onehot[subtype].encode(cse_data[subtype])
            dataset[subtype] = Dataset(self.params).train_test_split(
                oh_data[subtype])

        return dataset
