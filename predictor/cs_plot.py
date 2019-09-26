import numpy as np
from matplotlib import pyplot as plt
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
        Plot a candlestick diagram from a Dataframe. The colum names that
        contains the OHLC values can be specified as an array of strings
        to the arguments of the plot function, in case your columns are
        called any other way.
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


def plot_body_prediction(raw_prediction, pred_body_cs):
    # Plot the raw prediction from the NN
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(raw_prediction)
    winner_prediction = max(raw_prediction, key=abs)
    pos = np.where(raw_prediction == winner_prediction)[0][0]
    plt.plot(pos, winner_prediction, 'yo')
    plt.annotate(
        '{}={}'.format(pos, pred_body_cs[0]),
        xy=(pos, winner_prediction),
        xytext=(pos + 0.5, winner_prediction))
    plt.xticks(np.arange(0, len(raw_prediction), 1.0))
    ax.xaxis.label.set_size(6)


def plot_move_prediction(y, Y_pred, pred_move_cs, num_predictions,
                         pred_length):
    # find the position of the absmax mvalue in each of the arrays
    y_maxed = np.zeros(y.shape)
    for i in range(num_predictions):
        winner_prediction = max(Y_pred[i], key=abs)
        pos = np.where(Y_pred[i] == winner_prediction)[0][0]
        y_maxed[(i * pred_length) + pos] = winner_prediction

    # Plot the raw prediction from the NN
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(y)
    for i in range(len(y_maxed)):
        if y_maxed[i] != 0.0:
            plt.plot(i, y[i], 'yo')
            plt.annotate(
                '{}={}'.format(i, pred_move_cs[int(i / pred_length)]),
                xy=(i, y[i]),
                xytext=(i + 0.6, y[i]))
    plt.xticks(np.arange(0, len(y), 2.0))
    ax.xaxis.label.set_size(2)
    for vl in [i * pred_length for i in range(num_predictions + 1)]:
        plt.axvline(x=vl, linestyle=':', color='red')
    plt.show()


def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
