"""
Trend of the closing value: +1 if positive, -1 if negative
"""

from pandas import DataFrame
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import medfilt

from .base_indicators import *
from .indicator import Indicator


class gmf_mono(Indicator):
    """Gaussian Median Filter over diff values to return positive, negative
    or neutral trend (+1/-1/0)"""

    values: DataFrame = None

    # The name of the signal/indicator
    name = 'gmf_mono'
    # The columns that will be generated, and that must be saved/appended
    ix_columns = ['gmf_mono']

    def __init__(
            self,
            data,
            params,
            column_name='change',
            monotonic_window=5,
            mf_window=5,
            sigma=2,
    ):
        super(gmf_mono, self).__init__(data, params)
        self.params = params
        self.values = self.fit(column_name, monotonic_window, mf_window, sigma,
                               fill_na=True)

    def fit(self,
            column_name='change',
            monotonic_window=5,
            mf_window=5,
            sigma=2,
            fill_na: bool = True) -> DataFrame:
        """
        Compute the Gaussian1d of the Median Filter over the Price change
        of the close value. The result is a signal that when is detected to
        be monotonic positive, indicates positive trend for the actual stock
        value. The same applies for negative trends. When no monotonic
        pattern is identified, the signal returns a neutral value.

        Params
        ------

        column_name: The name of the column of the dataframe over which
            computing the GMF signal.
        monotonic_window: The window for which the monotonic behavior is
            computed. A value of, e.g.: 5, indicates that 5 consecutive
            increasing or decreasing values are needed to assign an increasing
            or decreasing behaviour of the signal.
        mf_window: The window size for the median filter to be applied over
            the output of the Gaussian filter
        sigma: The sigma value for the Gaussian1d.

        """
        assert column_name in self.data.columns, \
            f"A column with the {column_name} name must be present in the data"
        data = self.data.copy(deep=True)
        y = data[column_name]

        #
        # Start coding here.
        #
        gmf_values = pd.Series(
            gaussian_filter1d(
                medfilt(y, mf_window),
                sigma))
        in_signal = self.get_positive_periods(gmf_values, monotonic_window)
        out_signal = self.get_negative_periods(gmf_values, monotonic_window)
        gmf_signal = in_signal + out_signal

        self.values = pd.DataFrame(gmf_signal, columns=[self.name])
        if fill_na is True:
            self.values = self.values.replace(np.nan, 0.)

        return self.values

    @staticmethod
    def get_positive_periods(x, period_length):
        """This signal is 1 if it is monotonic positively growing over a given
        period_length"""

        def monotonic_positive(x):
            dx = np.diff(x)
            return np.all(dx >= 0)

        periods = x.rolling(period_length + 1)
        is_monotonic_positive = periods.apply(
            lambda period: monotonic_positive(period))
        return np.nan_to_num(is_monotonic_positive)

    @staticmethod
    def get_negative_periods(x, period_length):
        """This signal is 1 if it is monotonic negatively growing over a given
        period_length"""

        def monotonic_negative(x):
            dx = np.diff(x)
            return np.all(dx <= 0)

        periods = x.rolling(period_length + 1)
        is_monotonic_negative = periods.apply(lambda x: monotonic_negative(x))
        return np.nan_to_num(is_monotonic_negative) * -1.0
