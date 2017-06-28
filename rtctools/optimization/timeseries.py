import numpy as np


class Timeseries:
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
        if len(values) == 1:
            values = values[0]
        if hasattr(values, '__iter__'):
            self._values = np.array(values, dtype=np.float64, copy=True)
        else:
            self._values = np.full_like(times, values, dtype=np.float64)

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

    def __neg__(self):
        return self.__class__(self.times, -self.values)

    def __repr__(self):
        return 'Timeseries({}, {})'.format(self._times, self._values)
