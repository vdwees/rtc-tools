from typing import Union

import casadi as ca

import numpy as np


class Timeseries:
    """
    Time series object, bundling time stamps with values.
    """

    def __init__(self, times: np.ndarray, values: Union[np.ndarray, list, ca.DM]):
        """
        Create a new time series object.

        :param times:  Iterable of time stamps.
        :param values: Iterable of values.
        """
        self.__times = times
        if len(values) == 1:
            values = values[0]
        if hasattr(values, '__iter__'):
            self.__values = np.array(values, dtype=np.float64, copy=True)
        else:
            self.__values = np.full_like(times, values, dtype=np.float64)

    @property
    def times(self) -> np.ndarray:
        """
        Array of time stamps.
        """
        return self.__times

    @property
    def values(self) -> np.ndarray:
        """
        Array of values.
        """
        return self.__values

    def __neg__(self) -> 'Timeseries':
        return self.__class__(self.times, -self.values)

    def __repr__(self) -> str:
        return 'Timeseries({}, {})'.format(self.__times, self.__values)
