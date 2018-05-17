import bisect
import logging
import os
from datetime import timedelta

import numpy as np

import rtctools.data.csv as csv
from rtctools._internal.caching import cached

from .simulation_problem import SimulationProblem

logger = logging.getLogger("rtctools")


class CSVMixin(SimulationProblem):
    """
    Adds reading and writing of CSV timeseries and parameters to your simulation problem.

    During preprocessing, files named ``timeseries_import.csv``, ``initial_state.csv``,
    and ``parameters.csv`` are read from the ``input`` subfolder.

    During postprocessing, a file named ``timeseries_export.csv`` is written to the ``output`` subfolder.

    :cvar csv_delimiter:           Column delimiter used in CSV files.  Default is ``,``.
    :cvar csv_validate_timeseries: Check consistency of timeseries.  Default is ``True``.
    """

    #: Column delimiter used in CSV files
    csv_delimiter = ','

    #: Check consistency of timeseries
    csv_validate_timeseries = True

    # Default names for timeseries I/O
    timeseries_import_basename = 'timeseries_import'
    timeseries_export_basename = 'timeseries_export'

    def __init__(self, **kwargs):
        # Check arguments
        assert('input_folder' in kwargs)
        assert('output_folder' in kwargs)

        # Save arguments
        self.__input_folder = kwargs['input_folder']
        self.__output_folder = kwargs['output_folder']

        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

    def pre(self):
        # Call parent class first for default behaviour.
        super().pre()

        # Helper function to check if initial state array actually defines
        # only the initial state
        def check_initial_state_array(initial_state):
            """
            Check length of initial state array, throw exception when larger than 1.
            """
            if initial_state.shape:
                raise Exception(
                    'CSVMixin: Initial state file {} contains more than one row of data. '
                    'Please remove the data row(s) that do not describe the initial '
                    'state.'.format(os.path.join(self.__input_folder, 'initial_state.csv')))

        # Read CSV files
        _timeseries = csv.load(
            os.path.join(self.__input_folder, self.timeseries_import_basename + '.csv'),
            delimiter=self.csv_delimiter, with_time=True)
        self.__timeseries_times = _timeseries[_timeseries.dtype.names[0]]
        self.__timeseries = {
            key: np.asarray(_timeseries[key], dtype=np.float64) for key in _timeseries.dtype.names[1:]}

        logger.debug("CSVMixin: Read timeseries.")

        try:
            _parameters = csv.load(
                os.path.join(self.__input_folder, 'parameters.csv'),
                delimiter=self.csv_delimiter)
            logger.debug("CSVMixin: Read parameters.")
            self.__parameters = {
                key: float(_parameters[key]) for key in _parameters.dtype.names}
        except IOError:
            self.__parameters = {}

        try:
            _initial_state = csv.load(
                os.path.join(self.__input_folder, 'initial_state.csv'),
                delimiter=self.csv_delimiter)
            logger.debug("CSVMixin: Read initial state.")
            check_initial_state_array(_initial_state)
            self.__initial_state = {
                key: float(_initial_state[key]) for key in _initial_state.dtype.names}
        except IOError:
            self.__initial_state = {}

        # Check for collisions in __initial_state and __timeseries
        for collision in set(self.__initial_state) & set(self.__timeseries):
            if self.__initial_state[collision] == self.__timeseries[collision][0]:
                continue
            else:
                logger.warning(
                    'CSVMixin: Entry {} in initial_state.csv conflicts with '
                    'timeseries_import.csv'.format(collision))

        self.__timeseries_times_sec = self.__datetime_to_sec(self.__timeseries_times)

        # Timestamp check
        if self.csv_validate_timeseries:
            for i in range(len(self.__timeseries_times_sec) - 1):
                if self.__timeseries_times_sec[i] >= self.__timeseries_times_sec[i + 1]:
                    raise Exception(
                        'CSVMixin: Time stamps must be strictly increasing.')

        self.__dt = self.__timeseries_times_sec[1] - self.__timeseries_times_sec[0]

        # Check if the timeseries are truly equidistant
        if self.csv_validate_timeseries:
            for i in range(len(self.__timeseries_times_sec) - 1):
                if self.__timeseries_times_sec[i + 1] - self.__timeseries_times_sec[i] != self.__dt:
                    raise Exception(
                        'CSVMixin: Expecting equidistant timeseries, the time step '
                        'towards {} is not the same as the time step(s) before. '
                        'Set equidistant=False if this is intended.'.format(
                            self.__timeseries_times[i + 1]))

    def initialize(self, config_file=None):
        # Set up experiment
        self.setup_experiment(0, self.__timeseries_times_sec[-1], self.__dt)

        # Load parameters from parameter config
        self.__parameter_variables = set(self.get_parameter_variables())

        logger.debug("Model parameters are {}".format(self.__parameter_variables))

        for parameter, value in self.__parameters.items():
            if parameter in self.__parameter_variables:
                logger.debug("CSVMixin: Setting parameter {} = {}".format(parameter, value))
                self.set_var(parameter, value)

        # Load input variable names
        self.__input_variables = set(self.get_input_variables().keys())

        # Set input values
        for variable, timeseries in self.__timeseries.items():
            if variable in self.__input_variables:
                value = timeseries[0]
                if np.isfinite(value):
                    self.set_var(variable, value)

        logger.debug("Model inputs are {}".format(self.__input_variables))

        # Empty output
        self.__output_variables = self.get_output_variables()
        n_times = len(self.__timeseries_times_sec)
        self.__output = {
            variable: np.full(n_times, np.nan) for variable in self.__output_variables}

        # Call super, which will also initialize the model itself
        super().initialize(config_file)

        # Extract consistent t0 values
        for variable in self.__output_variables:
            self.__output[variable][0] = self.get_var(variable)

    def update(self, dt):
        # Time step
        if dt < 0:
            dt = self.__dt

        # Current time stamp
        t = self.get_current_time()

        # Get current time index
        t_idx = bisect.bisect_left(self.__timeseries_times_sec, t + dt)

        # Set input values
        for variable, timeseries in self.__timeseries.items():
            if variable in self.__input_variables:
                value = timeseries[t_idx]
                if np.isfinite(value):
                    self.set_var(variable, value)

        # Call super
        super().update(dt)

        # Extract results
        for variable in self.__output_variables:
            self.__output[variable][t_idx] = self.get_var(variable)

    def post(self):
        # Call parent class first for default behaviour.
        super().post()

        # Write output
        names = ['time'] + sorted(set(self.__output.keys()))
        formats = ['O'] + (len(names) - 1) * ['f8']
        dtype = {'names': names, 'formats': formats}
        data = np.zeros(len(self.__timeseries_times), dtype=dtype)
        data['time'] = self.__timeseries_times
        for variable, values in self.__output.items():
            data[variable] = values

        fname = os.path.join(self.__output_folder, self.timeseries_export_basename + '.csv')
        csv.save(fname, data, delimiter=self.csv_delimiter, with_time=True)

    def __datetime_to_sec(self, d):
        # Return the date/timestamps in seconds since t0.
        if hasattr(d, '__iter__'):
            return np.array([(t - self.__timeseries_times[0]).total_seconds() for t in d])
        else:
            return (d - self.__timeseries_times[0]).total_seconds()

    def __sec_to_datetime(self, s):
        # Return the date/timestamps in seconds since t0 as datetime objects.
        if hasattr(s, '__iter__'):
            return [self.__timeseries_times[0] + timedelta(seconds=t) for t in s]
        else:
            return self.__timeseries_times[0] + timedelta(seconds=s)

    @cached
    def parameters(self):
        """
        Return a dictionary of parameters, including parameters in parameters CSV files.

        :returns: Dictionary of parameters
        """
        # Call parent class first for default values.
        parameters = super().parameters()

        # Load parameters from parameter config
        parameters.update(self.__parameters)

        if logger.getEffectiveLevel() == logging.DEBUG:
            for parameter in self.__parameters:
                logger.debug("CSVMixin: Read parameter {} ".format(parameter))

        return parameters

    def times(self, variable=None):
        """
        Return a list of all the timesteps in seconds.

        :param variable: Variable name.

        :returns: List of all the timesteps in seconds.
        """
        return self.__timeseries_times_sec

    @cached
    def initial_state(self):
        """
        The initial state. Includes entries from parent classes and initial_state.csv

        :returns: A dictionary of variable names and initial state (t0) values.
        """
        # Call parent class first for default values.
        initial_state = super().initial_state()

        # Set of model vars that are allowed to have an initial state
        valid_model_vars = set(self.get_state_variables()) | set(self.get_input_variables())

        # Load initial states from __initial_state
        for variable, value in self.__initial_state.items():

            # Get the cannonical vars and signs
            canonical_var, sign = self.alias_relation.canonical_signed(variable)

            # Only store variables that are allowed to have an initial state
            if canonical_var in valid_model_vars:
                initial_state[canonical_var] = value * sign

                if logger.getEffectiveLevel() == logging.DEBUG:
                        logger.debug("CSVMixin: Read initial state {} = {}".format(variable, value))
            else:
                logger.warning("CSVMixin: In initial_state.csv, {} is not an input or state variable.".format(variable))
        return initial_state

    def timeseries_at(self, variable, t):
        """
        Return the value of a timeseries at the given time.

        :param variable: Variable name.
        :param t: Time.

        :returns: The interpolated value of the time series.

        :raises: KeyError
        """
        values = self.__timeseries[variable]
        t_idx = bisect.bisect_left(self.__timeseries_times_sec, t)
        if self.__timeseries_times_sec[t_idx] == t:
            return values[t_idx]
        else:
            return np.interp1d(t, self.__timeseries_times_sec, values)
