"""
Trend of the closing value: +1 if positive, -1 if negative
"""

from pandas import DataFrame

from base_indicators import *
from indicator import Indicator


class change(Indicator):
    """Difference with the previous value (actual diff, and percentage)"""

    values: DataFrame = None

    # The name of the signal/indicator
    name = 'change'
    # The columns that will be generated, and that must be saved/appended
    ix_columns = ['change', 'returns']

    def __init__(self, data, params, column_name='close'):
        super(change, self).__init__(data, params)
        self.params = params
        self.values = self.fit(column_name, fill_na=True)

    def fit(self, column_name='close', fill_na: bool = True) -> DataFrame:
        """ Compute the trend of the close value """
        assert column_name in self.data.columns, \
            f"A column with the {column_name} name must be present in the data"
        data = self.data.copy(deep=True)

        y = data[column_name]
        y_new = pd.concat([y.diff(), y.pct_change()], axis=1)
        y_new.columns = self.ix_columns

        self.values = y_new
        if fill_na is True:
            self.values = self.values.replace(np.nan, 0.)
            # Replace INF by the closest NON-INF value.
            returns = self.values[self.ix_columns[1]].values
            if np.isinf(returns).any():
                ind = np.where(np.isinf(returns))[0]
                returns[ind] = returns[ind - 1]
                self.values[self.ix_columns[1]] = returns

        return self.values
