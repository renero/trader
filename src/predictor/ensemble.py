import re
from os.path import splitext, basename

import pandas as pd
from pandas import DataFrame, Series

from predictor.cs_dictionary import CSDictionary
from utils.file_io import save_dataframe
from utils.logger import Logger


class Ensemble:
    num_preds = 0
    net_names = []

    def __init__(self, params: CSDictionary):
        self.params = params
        self.log: Logger = params.log

        self.log.info(
            'Generating ensemble with: {}'.format(self.params.input_file))
        self.ensemble_predictions(single_ensemble=self.params.ensemble)

    def ensemble_predictions(self, single_ensemble: bool):
        df = self.read_predictions_file()
        weights = self.compute_weights(df)
        if single_ensemble:
            json_predictions = pd.DataFrame(
                pd.read_json(self.params.json_prediction,
                             typ='series', orient='records')).T
            ensemble_data = self.compute_weighted_prediction(
                json_predictions,
                weights)
        else:
            ensemble_data = self.compute_weighted_prediction(df, weights)
        if self.params.save_predictions:
            self.save_ensemble(ensemble_data)
        else:
            if single_ensemble:
                to_show = ensemble_data[['w_avg']]
            else:
                to_show = ensemble_data[['actual', 'w_avg']]
            if self.params.predict:
                print(pd.DataFrame(
                    to_show.iloc[-1]).T.to_string())
            else:
                pd.set_option('display.max_rows', -1)
                to_show.T[0].to_json(self.params.json_forecast)
                self.log.info('Saved forecast: {}'.format(
                    self.params.json_forecast))
                print(to_show.to_string())

    def read_predictions_file(self) -> DataFrame:
        self.log.debug(
            'Reading predictions file: {}'.format(self.params.input_file))
        df = pd.read_csv(self.params.input_file,
                         delimiter=self.params.delimiter)
        if 'avg' in df.columns:
            self.num_preds = df.columns.get_loc('avg') - 1
        else:
            self.log.error('Column called <avg> not present in pred_ file')
        return df

    def compute_weights(self, preds: DataFrame) -> DataFrame:
        self.log.debug('Computing weights from different networks')
        # Take only the names of the networks
        from_position = list(preds.columns).index('actual') + 1
        self.net_names = preds.columns[from_position:self.num_preds + 1]
        self.log.debug('Network names: {}'.format(self.net_names))
        # Compute proportions
        proportions = preds.winner.value_counts()
        weights = pd.DataFrame({'proportion': proportions,
                                'weight': pd.Series(index=proportions.index)})
        weights.weight = weights.proportion / weights.proportion.sum()
        return weights

    def compute_weighted_prediction(self, df, weights):
        self.log.debug('Computing final weighted prediction')
        preds = df.copy(deep=True)

        def weighted_prediction(x: Series):  # , weights: DataFrame):
            w = weights.reindex(list(x.index)).weight
            x_w = x * w
            self.log.debug(
                'Weighting X·W = {}·{} = {}'.format(
                    x.values, w.values, x_w.values))
            self.log.debug('sum(X·W) = {}'.format(x_w.sum()))
            return x_w.sum()

        preds['w_avg'] = preds[self.net_names].apply(
            lambda x: weighted_prediction(x), axis=1)

        return preds

    def save_ensemble(self, preds: DataFrame):
        # Set the filename of the output file.
        if self.params.output is not None:
            new_filename = self.params.output
        else:
            current_filename = splitext(basename(self.params.input_file))[0]
            if re.search('^pred_', current_filename) is not None:
                new_filename = current_filename.replace('pred_', 'forecast_')
            else:
                new_filename = 'forecast_' + current_filename

        # Prepare the data frame to be saved.
        date_column = self.params.csv_dict['d']
        preds.reset_index(drop=True, inplace=True)
        preds.rename(columns={'w_avg': 'forecast'}, inplace=True)
        saved_file, _ = save_dataframe(
            new_filename,
            preds[[date_column, 'actual', 'forecast']].round(2),
            self.params.predictions_path,
            index=False)
        self.log.info('Saved forecast file: {}'.format(saved_file))
