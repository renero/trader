import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from pandas import Series


def trend_lines(ax1, trend, **kwargs):
    # plot lines where trend changes
    for x in trend.index[trend.green == 1].values:
        ax1.axvline(x, color='green', linestyle='--', alpha=0.2, **kwargs)
    for x in trend.index[trend.blue == 1].values:
        ax1.axvline(x, color='blue', linestyle='-.', alpha=0.2, **kwargs)


def plot_konkorde(data_series):
    data = data_series.copy(deep=True)
    data = data.reset_index(drop=True)
    k_size = data['verde'].shape[0]

    def trends(data: Series) -> Series:
        s = data.rolling(2).apply(
            lambda x: 1 if np.sign(x.iloc[0]) != np.sign(x.iloc[1]) else 0,
            raw=False)
        s.iloc[0] = 0.
        return s

    trend = pd.DataFrame(
        {'green': trends(data.verde), 'blue': trends(data.azul)})

    # plot stock price
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(211)
    ax1.plot(data.close, 'k', linewidth=0.8)
    plt.setp(ax1.get_xticklabels(), visible=False)
    trend_lines(ax1, trend)

    # plot green and blue
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.axhline(0, color='lightgrey')
    ax2.fill_between(range(k_size), data.verde,
                     facecolor='green', alpha=0.2)
    ax2.fill_between(range(k_size), data.azul,
                     facecolor='blue', alpha=0.2)
    ax2.plot(data.verde, 'g--', linewidth=0.6)
    ax2.plot(data.azul, 'b-.', linewidth=0.6)
    trend_lines(ax2, trend, linewidth=0.5)

    # finish job and leave, quitely...
    plt.tight_layout()
    plt.show()


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
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
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


def plot_marks(data: pd.DataFrame, signal: str, marks: str):
    # Conversion of dates to matplotlib numeric references
    inxval = mdates.date2num(data.index.to_pydatetime())

    fig, ax1 = plt.subplots()
    fig.set_size_inches(20, 6)

    g = pd.Series(data[marks])
    positives = g.where(g > 0.0).replace(np.nan, 0.0)
    negatives = g.where(g <= 0.0).replace(np.nan, 0.0)

    min, max = data[signal].min(), data[signal].max()
    marks_height = (max-min) / 25.0

    ax1.plot(data[signal], color="C0", alpha=0.8, linewidth=1.2)
    markerline, stemline, baseline = ax1.stem(
        inxval,
        positives * marks_height,
        markerfmt=" ",
        linefmt="C2-",
        basefmt="C7:",
        use_line_collection=True,
    )
    plt.setp(baseline, "linewidth", 0.2)
    plt.setp(stemline, "linewidth", 1)
    plt.setp(stemline, "alpha", 0.5)
    markerline, stemline, baseline = ax1.stem(
        inxval,
        negatives * marks_height,
        markerfmt=" ",
        linefmt="C3-",
        basefmt="C7:",
        use_line_collection=True,
    )
    plt.setp(baseline, "linewidth", 0.4)
    plt.setp(stemline, "linewidth", 1)
    plt.setp(stemline, "alpha", 0.5)
    plt.show()


