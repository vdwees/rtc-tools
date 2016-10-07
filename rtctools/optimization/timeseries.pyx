# cython: embedsignature=True

import numpy as np


class Timeseries(object):
    """
    Time series object, bundling time stamps with values.
    """

    def __init__(self, times, values):
        """
        Create a new time series object.

        :param times:  Iterable of time stamps.
        :param values: Iterable of values.
        """
        self._times = times
        self._values = np.copy(values)

    @property
    def times(self):
        """
        Array of time stamps.
        """
        return self._times

    @property
    def values(self):
        """
        Array of values.
        """
        return self._values

    def __repr__(self):
        return 'Timeseries({}, {})'.format(self._times, self._values)
