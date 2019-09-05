import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc


class CSPlot(object):

    _default_ohlc_names = ['Open', 'High', 'Low', 'Close']
    _open = _default_ohlc_names[0]
    _high = _default_ohlc_names[1]
    _low = _default_ohlc_names[2]
    _close = _default_ohlc_names[3]

    _default_width = 6
    _default_height = 4
    _default_style = 'dark_background'
    _default_color_up = '#77d879'
    _default_color_down = '#db3f3f'

    def __init__(self):
        pass

    @classmethod
    def candlesticks(cls, data, title=None, ohlc_names=_default_ohlc_names):
        """
        Plot a candlestick diagram from a Dataframe. The colum names that contains
        the OHLC values can be specified as an array of strings to the arguments of
        the plot function, in case your columns are called any other way.
        Arguments:
        - data: A dataframe with the open, high, low and close values in
                columns
        - title: Optional plot title. Default None
        - ohlc_names: The names of the columns in the dataframe that contain
                the values for the open, high, low and close.
        """
        plt.style.use(cls._default_style)
        fig, ax = plt.subplots(
            figsize=(cls._default_width, cls._default_height))
        ax.set_axisbelow(True)
        ax.set_title('{}'.format(title))
        ax.grid(color='#d7d7d7', linestyle='dashed', linewidth=0.5, axis='y')
        fig.subplots_adjust(bottom=0.2)
        index_field = list(map(int, data.index))
        candlestick_ohlc(
            ax,
            zip(index_field, data[ohlc_names[0]], data[ohlc_names[1]],
                data[ohlc_names[2]], data[ohlc_names[3]]),
            colorup=cls._default_color_up,
            colordown=cls._default_color_down,
            width=0.6)
        plt.setp(
            plt.gca().get_xticklabels(),
            rotation=45,
            horizontalalignment='right')
        plt.show()
        return plt
