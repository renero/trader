import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


class metrics:

    def __init__(self):
        pass

    @classmethod
    def fail(cls, y: float, y_prev: float, yhat: float) -> bool:
        return np.sign(y - y_prev) != np.sign(yhat - y_prev)

    @classmethod
    def plot_and_compare(cls, df: DataFrame) -> None:
        fails = 0
        plt.figure(figsize=(14, 8))
        for i in range(1, df.shape[0]):
            segment_color = "red"
            if i > 0:
                if not cls.fail(df.iloc[i].y, df.iloc[i - 1].y,
                                df.iloc[i].yhat):
                    segment_color = "green"
                else:
                    fails += 1

            plt.plot(
                [i - 1, i],
                [df.y.iloc[i - 1], df.y.iloc[i]],
                marker=".",
                linewidth=0.8,
                linestyle="--",
                alpha=0.4,
                color="black",
            )
            plt.plot(
                [i - 1, i],
                [df.y.iloc[i - 1], df.yhat.iloc[i]],
                linewidth=1.0,
                color=segment_color,
            )

        hits = df.shape[0] - fails
        hits_pct = 100 * (hits / df.shape[0])
        plt.title(
            f"""Trend hits ({hits}/{hits_pct:.2f}%) 
             and fails ({fails}/{100. - hits_pct:.2f}%)"""
        )
        plt.show()

    @classmethod
    def trend_accuracy(cls, y: np.ndarray, yhat: np.ndarray) -> float:
        fails = 0
        for i in range(1, y.shape[0]):
            if i > 0:
                if cls.fail(y[i], y[i - 1], yhat[i]):
                    fails += 1
        return (y.shape[0] - fails) / y.shape[0]

    @classmethod
    def trend_binary_accuracy(cls, y: np.ndarray, yhat: np.ndarray) -> float:
        fails = 0
        for i in range(1, y.shape[0]):
            if i > 0:
                if y[i] != yhat[i]:
                    fails += 1
        return (y.shape[0] - fails) / y.shape[0]

    @classmethod
    def mean_error(cls, df: DataFrame) -> float:
        cum_distance = 0.0
        for i in range(1, df.shape[0]):
            cum_distance += np.abs(df.y.iloc[i] - df.yhat.iloc[i])
        return cum_distance / df.shape[0]
