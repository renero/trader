"""
Trend of the closing value: +1 if positive, -1 if negative
"""

from pandas import DataFrame

from .base_indicators import *
from .indicator import Indicator


class trend(Indicator):
    values: DataFrame = None

    # The name of the signal/indicator
    name = 'trend'
    # The columns that will be generated, and that must be saved/appended
    ix_columns = ['trend']

    def __init__(self, data, params, column_name='close'):
        super(trend, self).__init__(data, params)
        self.params = params
        self.values = self.fit(column_name, fill_na=False)

    def fit(self, column_name='close', fill_na: bool = True) -> DataFrame:
        """ Compute the trend of the given column name value """
        assert column_name in self.data.columns, \
            f"A column with the {column_name} name must be present in the data"
        data = self.data.copy(deep=True)

        y = data[column_name].values
        y_trend = np.sign(y[1:] - y[:-1])
        y_trend = np.insert(y_trend, 0, 1., axis=0)
        trend_value = pd.Series(
            map(lambda x: 0. if x == -1 else 1., y_trend))

        self.values = pd.DataFrame(
            trend_value,
            columns=[f'{column_name}_{self.name}'])
        if fill_na is True:
            self.values = self.values.replace(np.nan, 0.)

        return self.values
