import bisect
import logging
from datetime import timedelta

import numpy as np

import rtctools.data.pi as pi
import rtctools.data.rtc as rtc
from rtctools._internal.alias_tools import AliasDict
from rtctools._internal.caching import cached

from .simulation_problem import SimulationProblem

logger = logging.getLogger("rtctools")


class PIMixin(SimulationProblem):
    """
    Adds `Delft-FEWS Published Interface
    <https://publicwiki.deltares.nl/display/FEWSDOC/The+Delft-Fews+Published+Interface>`_
    I/O to your simulation problem.

    During preprocessing, files named ``rtcDataConfig.xml``, ``timeseries_import.xml``,  and``rtcParameterConfig.xml``
    are read from the ``input`` subfolder.  ``rtcDataConfig.xml`` maps
    tuples of FEWS identifiers, including location and parameter ID, to RTC-Tools time series identifiers.

    During postprocessing, a file named ``timeseries_export.xml`` is written to the ``output`` subfolder.

    :cvar pi_binary_timeseries: Whether to use PI binary timeseries format.  Default is ``False``.
    :cvar pi_parameter_config_basenames:
        List of parameter config file basenames to read. Default is [``rtcParameterConfig``].
    :cvar pi_check_for_duplicate_parameters: Check if duplicate parameters are read. Default is ``False``.
    :cvar pi_validate_timeseries: Check consistency of timeseries.  Default is ``True``.
    """

    #: Whether to use PI binary timeseries format
    pi_binary_timeseries = False

    #: Location of rtcParameterConfig files
    pi_parameter_config_basenames = ['rtcParameterConfig']

    #: Check consistency of timeseries
    pi_validate_timeseries = True

    #: Check for duplicate parameters
    pi_check_for_duplicate_parameters = True

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

        # Load rtcDataConfig.xml.  We assume this file does not change over the
        # life time of this object.
        self.__data_config = rtc.DataConfig(self.__input_folder)

        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

    def pre(self):
        # Call parent class first for default behaviour.
        super().pre()

        # rtcParameterConfig
        self.__parameter_config = []
        try:
            for pi_parameter_config_basename in self.pi_parameter_config_basenames:
                self.__parameter_config.append(pi.ParameterConfig(
                    self.__input_folder, pi_parameter_config_basename))
        except FileNotFoundError:
            raise FileNotFoundError(
                "PIMixin: {}.xml not found in {}.".format(pi_parameter_config_basename, self.__input_folder))

        # Make a parameters dict for later access
        self.__parameters = {}
        for parameter_config in self.__parameter_config:
            for location_id, model_id, parameter_id, value in parameter_config:
                try:
                    parameter = self.__data_config.parameter(parameter_id, location_id, model_id)
                except KeyError:
                    parameter = parameter_id
                self.__parameters[parameter] = value

        try:
            self.__timeseries_import = pi.Timeseries(
                self.__data_config, self.__input_folder, self.timeseries_import_basename,
                binary=self.pi_binary_timeseries, pi_validate_times=self.pi_validate_timeseries)
        except FileNotFoundError:
            raise FileNotFoundError('PIMixin: {}.xml not found in {}'.format(
                self.timeseries_import_basename, self.__input_folder))

        self.__timeseries_export = pi.Timeseries(
            self.__data_config, self.__output_folder, self.timeseries_export_basename,
            binary=self.pi_binary_timeseries, pi_validate_times=False, make_new_file=True)

        # Convert timeseries timestamps to seconds since t0 for internal use
        self.__timeseries_import_times = self.__datetime_to_sec(
            self.__timeseries_import.times)

        # Timestamp check
        if self.pi_validate_timeseries:
            for i in range(len(self.__timeseries_import_times) - 1):
                if self.__timeseries_import_times[i] >= self.__timeseries_import_times[i + 1]:
                    raise ValueError(
                        'PIMixin: Time stamps must be strictly increasing.')

        # Check if the timeseries are equidistant
        self.__dt = self.__timeseries_import_times[1] - self.__timeseries_import_times[0]
        if self.pi_validate_timeseries:
            for i in range(len(self.__timeseries_import_times) - 1):
                if self.__timeseries_import_times[i + 1] - self.__timeseries_import_times[i] != self.__dt:
                    raise ValueError(
                        'PIMixin: Expecting equidistant timeseries, the time step '
                        'towards {} is not the same as the time step(s) before. Set '
                        'unit to nonequidistant if this is intended.'.format(
                            self.__timeseries_import.times[i + 1]))

        # Stick timeseries into an AliasDict
        self.__timeseries_import_dict = AliasDict(self.alias_relation)
        debug = logger.getEffectiveLevel() == logging.DEBUG
        for variable, values in self.__timeseries_import.items():
            if debug and self.__timeseries_import_dict.get(variable, None) is not None:
                logger.debug('PIMixin: Timeseries {} replaced another aliased timeseries.'.format(variable))
            self.__timeseries_import_dict[variable] = values

    def initialize(self, config_file=None):
        # Set up experiment
        self.setup_experiment(0, self.__timeseries_import_times[-1], self.__dt)

        # Load parameters from parameter config
        self.__parameter_variables = set(self.get_parameter_variables())

        logger.debug("Model parameters are {}".format(self.__parameter_variables))

        for parameter, value in self.__parameters.items():
            if parameter in self.__parameter_variables:
                logger.debug("PIMixin: Setting parameter {} = {}".format(parameter, value))
                self.set_var(parameter, value)

        # Load input variable names
        self.__input_variables = set(self.get_input_variables().keys())

        # Set input values
        for variable in self.__input_variables:
            value = self.__timeseries_import_dict[variable][self.__timeseries_import.forecast_index]
            if np.isfinite(value):
                self.set_var(variable, value)
            else:
                logger.debug('PIMixin: Found bad value {} at index [{}] in timeseries aliased to input {}'.format(
                    value, self.__timeseries_import.forecast_index, variable))

        logger.debug("Model inputs are {}".format(self.__input_variables))

        # Empty output
        self.__output_variables = self.get_output_variables()
        n_times = len(self.__timeseries_import_times)
        self.__output = AliasDict(self.alias_relation)
        self.__output.update({variable: np.full(n_times, np.nan) for variable in self.__output_variables})

        # Call super, which will also initialize the model itself
        super().initialize(config_file)

        # Extract consistent t0 values
        for variable in self.__output_variables:
            self.__output[variable][self.__timeseries_import.forecast_index] = self.get_var(variable)

    def update(self, dt):
        # Time step
        if dt < 0:
            dt = self.__dt

        # Current time stamp
        t = self.get_current_time()

        # Get current time index
        t_idx = bisect.bisect_left(self.__timeseries_import_times, t + dt)

        # Set input values
        for variable in self.__input_variables:
            value = self.__timeseries_import_dict[variable][t_idx]
            if np.isfinite(value):
                self.set_var(variable, value)
            else:
                logger.debug('PIMixin: Found bad value {} at index [{}] in timeseries aliased to input {}'.format(
                    value, t_idx, variable))

        # Call super
        super().update(dt)

        # Extract results
        for variable in self.__output_variables:
            self.__output[variable][t_idx] = self.get_var(variable)

    def post(self):
        # Call parent class first for default behaviour.
        super().post()

        # Start of write output
        # Write the time range for the export file.
        self.__timeseries_export.times = self.__timeseries_import.times[self.__timeseries_import.forecast_index:]

        # Write other time settings
        self.__timeseries_export.forecast_datetime = self.__timeseries_import.forecast_datetime
        self.__timeseries_export.dt = self.__timeseries_import.dt
        self.__timeseries_export.timezone = self.__timeseries_import.timezone

        # Write the ensemble properties for the export file.
        self.__timeseries_export.ensemble_size = 1
        self.__timeseries_export.contains_ensemble = self.__timeseries_import.contains_ensemble

        # For all variables that are output variables the values are
        # extracted from the results.
        for variable in self.__output_variables:
            values = self.__output[variable]
            # Check if ID mapping is present
            try:
                self.__data_config.pi_variable_ids(variable)
            except KeyError:
                logger.debug(
                    'PIMixin: variable {} has no mapping defined in rtcDataConfig '
                    'so cannot be added to the output file.'.format(variable))
                continue

            # Add series to output file
            self.__timeseries_export.set(variable, values, unit=self.__timeseries_import.get_unit(variable))

        # Write output file to disk
        self.__timeseries_export.write()

    def __datetime_to_sec(self, d):
        # Return the date/timestamps in seconds since t0.
        if hasattr(d, '__iter__'):
            return np.array([(t - self.__timeseries_import.forecast_datetime).total_seconds() for t in d])
        else:
            return (d - self.__timeseries_import.forecast_datetime).total_seconds()

    def __sec_to_datetime(self, s):
        # Return the date/timestamps in seconds since t0 as datetime objects.
        if hasattr(s, '__iter__'):
            return [self.__timeseries_import.forecast_datetime + timedelta(seconds=t) for t in s]
        else:
            return self.__timeseries_import.forecast_datetime + timedelta(seconds=s)

    @cached
    def parameters(self):
        """
        Return a dictionary of parameters, including parameters in PI Parameter Config XML files.

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

    @cached
    def times(self, variable=None):
        """
        Return a list of all the timesteps in seconds.

        :param variable: Variable name.

        :returns: A list of all the timesteps in seconds.
        """
        return self.__timeseries_import_times[self.__timeseries_import.forecast_index:]

    def timeseries_at(self, variable, t):
        """
        Return the value of a time series at the given time.

        :param variable: Variable name.
        :param t: Time.

        :returns: The interpolated value of the time series.

        :raises: KeyError
        """
        values = self.__timeseries_import_dict[variable]
        t_idx = bisect.bisect_left(self.__timeseries_import_times, t)
        if self.__timeseries_import_times[t_idx] == t:
            return values[t_idx]
        else:
            return np.interp1d(t, self.__timeseries_import_times, values)

    @property
    def timeseries_import(self):
        """
        :class:`pi.Timeseries` object containing the input data.
        """
        return self.__timeseries_import

    @property
    def timeseries_import_times(self):
        """
        List of time stamps for which input data is specified.

        The time stamps are in seconds since t0, and may be negative.
        """
        return self.__timeseries_import_times

    @property
    def timeseries_export(self):
        """
        :class:`pi.Timeseries` object for holding the output data.
        """
        return self.__timeseries_export

    def set_timeseries(self, variable, values, output=True, check_consistency=True, unit=None):
        if check_consistency:
            if len(self.times()) != len(values):
                raise ValueError(
                    'PIMixin: Trying to set/append values {} with a different '
                    'length than the forecast length. Please make sure the '
                    'values cover forecastDate through endDate with timestep {}.'.format(
                        variable, self.__timeseries_import.dt))

        if unit is None:
            unit = self.__timeseries_import.get_unit(variable)

        if output:
            try:
                self.__data_config.pi_variable_ids(variable)
            except KeyError:
                logger.debug(
                    'PIMixin: variable {} has no mapping defined in rtcDataConfig '
                    'so cannot be added to the output file.'.format(variable))
            else:
                self.__timeseries_export.set(variable, values, unit=unit)

        self.__timeseries_import.set(variable, values, unit=unit)
        self.__timeseries_import_dict[variable] = values

    def get_timeseries(self, variable):
        return self.__timeseries_import_dict[variable]

    def extract_results(self):
        """
        Extracts the results of output

        :returns: An AliasDict of output variables and results array format.
        """
        return self.__output
