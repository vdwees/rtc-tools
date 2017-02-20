# cython: embedsignature=True

from datetime import timedelta
import numpy as np
import logging
import bisect

import rtctools.data.rtc as rtc
import rtctools.data.pi as pi

from simulation_problem import SimulationProblem

logger = logging.getLogger("rtctools")


class PIMixin(SimulationProblem):
    """
    Adds `Delft-FEWS Published Interface <https://publicwiki.deltares.nl/display/FEWSDOC/The+Delft-Fews+Published+Interface>`_ I/O to your simulation problem.

    During preprocessing, files named ``rtcDataConfig.xml``, ``timeseries_import.xml``,  and``rtcParameterConfig.xml``
    are read from the ``input`` subfolder.  ``rtcDataConfig.xml`` maps
    tuples of FEWS identifiers, including location and parameter ID, to RTC-Tools time series identifiers.

    During postprocessing, a file named ``timeseries_export.xml`` is written to the ``output`` subfolder.

    :cvar pi_binary_timeseries: Whether to use PI binary timeseries format.  Default is ``False``.
    :cvar pi_parameter_config_basenames: List of parameter config file basenames to read. Default is [``rtcParameterConfig``].
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

    def __init__(self, **kwargs):
        # Check arguments
        assert('input_folder' in kwargs)
        assert('output_folder' in kwargs)

        # Save arguments
        self._input_folder = kwargs['input_folder']
        self._output_folder = kwargs['output_folder']

        # Load rtcDataConfig.xml.  We assume this file does not change over the
        # life time of this object.
        self._data_config = rtc.DataConfig(self._input_folder)

        # Call parent class first for default behaviour.
        super(PIMixin, self).__init__(**kwargs)

    def pre(self):
        # Call parent class first for default behaviour.
        super(PIMixin, self).pre()

        # rtcParameterConfig
        self._parameter_config = []
        try:
            for pi_parameter_config_basename in self.pi_parameter_config_basenames:
                self._parameter_config.append(pi.ParameterConfig(
                    self._input_folder, pi_parameter_config_basename))
        except IOError:
            raise Exception(
                "PI: {}.xml not found in {}.".format(pi_parameter_config_basename, self._input_folder))

        # timeseries_{import,export}.xml. rtcDataConfig can override (if not
        # falsy)
        basename_import = self._data_config._basename_import or 'timeseries_import'
        basename_export = self._data_config._basename_export or 'timeseries_export'

        try:
            self._timeseries_import = pi.Timeseries(
                self._data_config, self._input_folder, basename_import, binary=self.pi_binary_timeseries, pi_validate_times=self.pi_validate_timeseries)
        except IOError:
            raise Exception("PI: {}.xml not found in {}.".format(
                basename_import, self._input_folder))

        self._timeseries_export = pi.Timeseries(
            self._data_config, self._output_folder, basename_export, binary=self.pi_binary_timeseries, pi_validate_times=False, make_new_file=True)

        # Convert timeseries timestamps to seconds since t0 for internal use
        self._timeseries_import_times = self._datetime_to_sec(
            self._timeseries_import.times)

        # Timestamp check
        if self.pi_validate_timeseries:
            for i in range(len(self._timeseries_import_times) - 1):
                if self._timeseries_import_times[i] >= self._timeseries_import_times[i + 1]:
                    raise Exception(
                        'PIMixin: Time stamps must be strictly increasing.')

        # Check if the timeseries are equidistant
        self._dt = self._timeseries_import_times[1] - self._timeseries_import_times[0]
        if self.pi_validate_timeseries:
            for i in range(len(self._timeseries_import_times) - 1):
                if self._timeseries_import_times[i + 1] - self._timeseries_import_times[i] != self._dt:
                    raise Exception('PIMixin: Expecting equidistant timeseries, the time step towards {} is not the same as the time step(s) before. Set unit to nonequidistant if this is intended.'.format(
                        self._timeseries_import.times[i + 1]))

    def initialize(self, config_file=None):
        # Set up experiment
        self.setup_experiment(0, self._timeseries_import_times[-1], self._dt)

        # Load parameters from parameter config
        self._parameter_variables = set(self.get_parameter_variables())

        logger.debug("Model parameters are {}".format(self._parameter_variables))

        for parameter_config in self._parameter_config:
            for location_id, model_id, parameter_id, value in parameter_config:
                try:
                    parameter = self._data_config.parameter(parameter_id, location_id, model_id)
                except KeyError:
                    parameter = parameter_id

                if parameter in self._parameter_variables:
                    logger.debug("Setting parameter {} = {}".format(parameter, value))

                    self.set_var(parameter, value)

        # Load input variable names
        self._input_variables = set(self.get_input_variables().keys())

        logger.debug("Model inputs are {}".format(self._input_variables))

        # Set initial input values
        for variable, timeseries in self._timeseries_import.iteritems():
            if variable in self._input_variables:
                value = timeseries[self._timeseries_import._forecast_index]
                if np.isfinite(value):
                    self.set_var(variable, value)

        # Empty output
        self._output_variables = set(self.get_output_variables().keys())
        n_times = len(self._timeseries_import_times)
        self._output = {variable : np.full(n_times, np.nan) for variable in self._output_variables}

        # Call super, which will also initialize the model itself
        super(PIMixin, self).initialize(config_file)

        # Extract consistent t0 values
        for variable in self._output_variables:
            self._output[variable][self._timeseries_import._forecast_index] = self.get_var(variable)

    def update(self, dt):
        # Current time stamp
        t = self.get_current_time()   

        # Get current time index
        t_idx = bisect.bisect_left(self._timeseries_import_times, t)  

        # Set input values
        for variable, timeseries in self._timeseries_import.iteritems():
            if variable in self._input_variables:
                value = timeseries[t_idx]
                if np.isfinite(value):
                    self.set_var(variable, value)

        # Call super
        super(PIMixin, self).update(self._dt)

        # Extract results
        for variable in self._output_variables:
            self._output[variable][t_idx + 1] = self.get_var(variable)

    def post(self):
        # Call parent class first for default behaviour.
        super(PIMixin, self).post()

        # Start of write output
        # Write the time range for the export file.
        self._timeseries_export._times = self._timeseries_import.times[self._timeseries_import.forecast_index:]

        # Write other time settings
        self._timeseries_export._start_datetime = self._timeseries_import._forecast_datetime
        self._timeseries_export._end_datetime  = self._timeseries_import._end_datetime
        self._timeseries_export._forecast_datetime  = self._timeseries_import._forecast_datetime
        self._timeseries_export._dt = self._timeseries_import._dt
        self._timeseries_export._timezone = self._timeseries_import._timezone

        # Write the ensemble properties for the export file.
        self._timeseries_export._ensemble_size = 1
        self._timeseries_export._contains_ensemble = self._timeseries_import.contains_ensemble
        while self._timeseries_export._ensemble_size > len(self._timeseries_export._values):
            self._timeseries_export._values.append({})

        # Transfer units from import timeseries
        self._timeseries_export._units = self._timeseries_import._units

        # For all variables that are output variables the values are
        # extracted from the results.
        for key, values in self._output.iteritems():
            # Check if ID mapping is present
            try:
                location_parameter_id = self._timeseries_export._data_config.pi_variable_ids(key)
            except KeyError:
                logger.debug('PIMixIn: variable {} has no mapping defined in rtcDataConfig so cannot be added to the output file.'.format(key))
                continue

            # Add series to output file
            self._timeseries_export.set(key, values)

        # Write output file to disk
        self._timeseries_export.write()

    def _datetime_to_sec(self, d):
        # Return the date/timestamps in seconds since t0.
        if hasattr(d, '__iter__'):
            return np.array([(t - self._timeseries_import.forecast_datetime).total_seconds() for t in d])
        else:
            return (d - self._timeseries_import.forecast_datetime).total_seconds()

    def _sec_to_datetime(self, s):
        # Return the date/timestamps in seconds since t0 as datetime objects.
        if hasattr(s, '__iter__'):
            return [self._timeseries_import.forecast_datetime + timedelta(seconds=t) for t in s]
        else:
            return self._timeseries_import.forecast_datetime + timedelta(seconds=s)
