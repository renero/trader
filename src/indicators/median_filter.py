"""
Trend of the closing value: +1 if positive, -1 if negative
"""

from pandas import DataFrame
from scipy.signal import medfilt

from .base_indicators import *
from .indicator import Indicator


class median_filter(Indicator):
    values: DataFrame = None

    # The name of the signal/indicator
    name = 'med_filter'
    # The columns that will be generated, and that must be saved/appended
    ix_columns = ['med_filter']

    def __init__(self,
                 data,
                 params,
                 column_name='close',
                 window=5,
                 fill_na=False):
        super(median_filter, self).__init__(data, params)
        self.params = params
        self.values = self.fit(column_name, window, fill_na)

    def fit(self,
            column_name='close',
            window=5,
            fill_na: bool = True) -> DataFrame:
        """ Compute the median filter of the close value """
        assert column_name in self.data.columns, \
            f"A column with the {column_name} name must be present in the data"
        if window % 2 == 0:
            window += 1

        data = self.data.copy(deep=True)
        mf = medfilt(data[column_name].values, window)

        self.values = pd.DataFrame(mf, columns=[self.name])
        if fill_na is True:
            self.values = self.values.replace(np.nan, 0.)

        return self.values
