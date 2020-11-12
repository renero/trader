from itertools import chain
from typing import Union

from cvxopt import solvers, matrix, spmatrix
from pandas import DataFrame

from .base_indicators import *
from .indicator import Indicator

solvers.options['show_progress'] = 0


class L1tf:
    """
    Original code from: https://github.com/elsonidoq/py-l1tf
    """

    data = None
    M = None
    m = None
    denom = None
    t = None
    values = None

    def __init__(self):
        return

    def fit(self, data: Union[pd.Series, np.array]) -> "L1tf":
        """
        :param data: A time series signal, either a numpy array or pandas Series

        :return: An L1tf object.
        """
        self.data = data
        self.m = float(data.min())
        self.M = float(data.max())
        self.denom = self.M - self.m
        # if denom == 0, data is constant
        self.t = (data - self.m) / (1 if self.denom == 0 else self.denom)

        if isinstance(data, np.ndarray):
            self.values = matrix(self.t)
        elif isinstance(data, pd.Series):
            self.values = matrix(self.t.values[:])
        else:
            raise ValueError("Wrong type for data")

        return self

    def transform(self, delta: float = 0.1) -> Union[pd.Series, np.array]:
        """
        :param delta: Strength of regularization

        :return: The filtered series
        """
        self.values = self._l1tf(delta)
        self.values = self.values * (self.M - self.m) + self.m
        if isinstance(self.data, np.ndarray):
            self.values = np.asarray(self.values).squeeze()
        elif isinstance(self.data, pd.Series):
            self.values = pd.Series(self.values, index=self.data.index,
                                    name=self.data.name)

        return self.values

    def fit_transform(self, data: Union[pd.Series, np.array],
                      delta: float = 0.1) -> Union[pd.Series, np.array]:
        return self.fit(data).transform(delta)

    def _l1tf(self, delta):
        """
        minimize    (1/2) * ||x-corr||_2^2 + delta * sum(y)
        subject to  -y <= D*x <= y
        Variables x (n), y (n-2).

        :param x:
        :return:
        """
        n = self.values.size[0]
        m = n - 2
        D = self._get_second_derivative_matrix(n)
        P = D * D.T
        q = -D * self.values
        G = spmatrix([], [], [], (2 * m, m))
        G[:m, :m] = spmatrix(1.0, range(m), range(m))
        G[m:, :m] = -spmatrix(1.0, range(m), range(m))
        h = matrix(delta, (2 * m, 1), tc='d')
        res = solvers.qp(P, q, G, h)
        return self.values - D.T * res['x']

    @staticmethod
    def _get_second_derivative_matrix(n):
        """
        :param n: The size of the time series

        :return: A matrix D such that if x.size == (n,1), D * x is the second
        derivate of x
        """
        m = n - 2
        D = spmatrix(list(chain(*[[1, -2, 1]] * m)),
                     list(chain(*[[i] * 3 for i in range(m)])),
                     list(chain(*[[i, i + 1, i + 2] for i in range(m)])))
        return D


class l1tf(Indicator):

    values: DataFrame = None

    # The name of the signal/indicator
    name = 'l1tf'
    # The columns that will be generated, and that must be saved/appended
    ix_columns = ['l1tf']

    def __init__(self, data, params, column_name='close'):
        super(l1tf, self).__init__(data, params)
        self.params = params
        self.values = self.fit(column_name, fill_na=False)

    def fit(self, column_name='close', delta: float = 0.1, fill_na: bool = True) -> DataFrame:
        """ Compute the trend of the given column name value """
        assert column_name in self.data.columns, \
            f"A column with the {column_name} name must be present in the data"
        data = self.data.copy(deep=True)

        y = data[column_name].values
        l1tf_value = L1tf().fit_transform(y, delta)

        self.values = pd.DataFrame(
            l1tf_value,
            columns=[f'{column_name}_{self.name}'])
        if fill_na is True:
            self.values = self.values.replace(np.nan, 0.)

        return self.values
