# cython: embedsignature=True

from datetime import timedelta
from casadi import MX
import numpy as np
import logging

import rtctools.data.rtc as rtc
import rtctools.data.pi as pi

from optimization_problem import OptimizationProblem
from timeseries import Timeseries

logger = logging.getLogger("rtctools")


class PIMixin(OptimizationProblem):
    """
    Adds `Delft-FEWS Published Interface <https://publicwiki.deltares.nl/display/FEWSDOC/The+Delft-Fews+Published+Interface>`_ I/O to your optimization problem.

    During preprocessing, files named ``rtcDataConfig.xml``, ``timeseries_import.xml``, ``rtcParameterConfig.xml``,
    and ``rtcParameterConfig_Numerical.xml`` are read from the ``input`` subfolder.  ``rtcDataConfig.xml`` maps
    tuples of FEWS identifiers, including location and parameter ID, to RTC-Tools time series identifiers.

    During postprocessing, a file named ``timeseries_export.xml`` is written to the ``output`` subfolder.

    :cvar pi_binary_timeseries:   Whether to use PI binary timeseries format.  Default is ``False``.
    :cvar pi_parameter_group:     Group of model parameters in rtcParameterConfig files.  Default is ``parameters``.
    :cvar pi_parameter_model:     Model ID of model parameters in rtcParameterConfig files.  Default is ``Model``.
    :cvar pi_solver_group:        Group of solver parameters in rtcParameterConfig files.  Default is ``solver``.
    :cvar pi_solver_model:        Model ID of solver parameters in rtcParameterConfig files.  Default is ``Solver``.
    :cvar pi_validate_timeseries: Check consistency of timeseries.  Default is ``True``.
    """

    #: Whether to use PI binary timeseries format
    pi_binary_timeseries = False

    #: Location of model parameters in rtcParameterConfig files
    pi_parameter_group = 'parameters'
    pi_parameter_model = 'Model'

    #: Location of solver parameters in rtcParameterConfig files
    pi_solver_group = 'solver'
    pi_solver_model = 'Solver'

    #: Check consistency of timeseries
    pi_validate_timeseries = True

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

        # Additional output variables
        self._output_timeseries = set()

        # Call parent class first for default behaviour.
        super(PIMixin, self).__init__(**kwargs)

    def pre(self):
        # Call parent class first for default behaviour.
        super(PIMixin, self).pre()

        # rtcParameterConfig.xml
        try:
            self._parameter_config = pi.ParameterConfig(
                self._input_folder, 'rtcParameterConfig')
        except IOError:
            raise Exception(
                "PI: rtcParameterConfig.xml not found in {}.".format(self._input_folder))

        try:
            self._parameter_config_numerical = pi.ParameterConfig(
                self._input_folder, 'rtcParameterConfig_Numerical')
        except IOError:
            self._parameter_config_numerical = None

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
            self._data_config, self._output_folder, basename_export, binary=self.pi_binary_timeseries, pi_validate_times=False)

        # Convert timeseries timestamps to seconds since t0 for internal use
        self._timeseries_import_times = self._datetime_to_sec(
            self._timeseries_import.times)

        # Timestamp check
        if self.pi_validate_timeseries:
            for i in range(len(self._timeseries_import_times) - 1):
                if self._timeseries_import_times[i] >= self._timeseries_import_times[i + 1]:
                    raise Exception(
                        'PIMixin: Time stamps must be strictly increasing.')

        if self.equidistant:
            # Check if the timeseries are truly equidistant
            if self.pi_validate_timeseries:
                dt = self._timeseries_import_times[
                    1] - self._timeseries_import_times[0]
                for i in range(len(self._timeseries_import_times) - 1):
                    if self._timeseries_import_times[i + 1] - self._timeseries_import_times[i] != dt:
                        raise Exception('PIMixin: Expecting equidistant timeseries, the time step towards {} is not the same as the time step(s) before. Set unit to nonequidistant if this is intended.'.format(
                            self._timeseries_import.times[i + 1]))

    def times(self, variable=None):
        return self._timeseries_import_times[self._timeseries_import.forecast_index:]

    @property
    def equidistant(self):
        if self._timeseries_import._dt:
            return True
        else:
            return False

    def solver_options(self):
        # Call parent
        options = super(PIMixin, self).solver_options()

        # Only do this if we have a rtcParameterConfig_Numerical
        if self._parameter_config_numerical == None:
            return options

        # Load solver options from parameter config
        for option, value in self._parameter_config_numerical:
            options[option] = value

        # Done
        return options

    def parameters(self, ensemble_member):
        # Call parent class first for default values.
        parameters = super(PIMixin, self).parameters(ensemble_member)

        # Load parameters from parameter config
        for parameter, value in self._parameter_config:
            parameters[parameter] = value

        # Done
        return parameters

    def constant_inputs(self, ensemble_member):
        # Call parent class first for default values.
        constant_inputs = super(PIMixin, self).constant_inputs(ensemble_member)

        # Load bounds from timeseries
        for variable in self.dae_variables['constant_inputs']:
            for alias in self.variable_aliases(variable.getName()):
                try:
                    constant_inputs[variable.getName()] = Timeseries(
                        self._timeseries_import_times, alias.sign * self._timeseries_import.get(alias.name, ensemble_member=ensemble_member))
                    logger.debug("Read constant input {} from {}".format(
                        variable.getName(), alias.name))
                    break
                except KeyError:
                    continue
        return constant_inputs

    def bounds(self):
        # Call parent class first for default values.
        bounds = super(PIMixin, self).bounds()

        # Load bounds from timeseries
        for variable in self.dae_variables['free_variables']:
            m, M = None, None
            for alias in self.variable_aliases(variable.getName()):
                try:
                    timeseries_id = self.min_timeseries_id(alias.name)
                    m = alias.sign * self._timeseries_import.get(timeseries_id, ensemble_member=0)[
                        self._timeseries_import.forecast_index:]
                    logger.debug("Read lower bound for variable {} from {}".format(
                        variable.getName(), timeseries_id))
                except KeyError:
                    pass

                try:
                    timeseries_id = self.max_timeseries_id(alias.name)
                    M = alias.sign * self._timeseries_import.get(timeseries_id, ensemble_member=0)[
                        self._timeseries_import.forecast_index:]
                    logger.debug("Read upper bound for variable {} from {}".format(
                        variable.getName(), timeseries_id))
                except KeyError:
                    pass

                if m != None and M != None:
                    break

            # Replace NaN with +/- inf, and create Timeseries objects
            if m != None:
                m[np.isnan(m)] = np.finfo(m.dtype).min
                m = Timeseries(self._timeseries_import_times[
                               self._timeseries_import.forecast_index:], m)
            if M != None:
                M[np.isnan(M)] = np.finfo(M.dtype).max
                M = Timeseries(self._timeseries_import_times[
                               self._timeseries_import.forecast_index:], M)

            # Store
            if m != None or M != None:
                bounds[variable.getName()] = (m, M)
        return bounds

    def history(self, ensemble_member):
        # Load history
        history = {}
        for state in self.dae_variables['states'] + self.dae_variables['algebraics'] + self.dae_variables['control_inputs'] + self.dae_variables['constant_inputs']:
            for alias in self.variable_aliases(state.getName()):
                try:
                    history[state.getName()] = Timeseries(self._timeseries_import_times[:self._timeseries_import.forecast_index + 1], alias.sign *
                                                          self._timeseries_import.get(alias.name, ensemble_member=ensemble_member)[:self._timeseries_import.forecast_index + 1])
                    logger.debug("Read history for state {} from {}".format(
                        state.getName(), alias.name))
                    break
                except KeyError:
                    continue
        return history

    @property
    def initial_time(self):
        return 0.0

    def initial_state(self, ensemble_member):
        history = self.history(ensemble_member)
        return {variable: timeseries.values[-1] for variable, timeseries in history.iteritems()}

    def seed(self, ensemble_member):
        # Call parent class first for default values.
        seed = super(PIMixin, self).seed(ensemble_member)

        # Load seeds
        for variable in self.dae_variables['free_variables']:
            for alias in self.variable_aliases(variable.getName()):
                try:
                    s = Timeseries(self._timeseries_import_times, alias.sign *
                                   self._timeseries_import.get(alias.name, ensemble_member=ensemble_member))
                    logger.debug("Seeded free variable {} from {}".format(
                        variable.getName(), alias.name))
                    # A seeding of NaN means no seeding
                    s.values[np.isnan(s.values)] = 0.0
                    seed[variable.getName()] = s
                    break
                except KeyError:
                    continue
        return seed

    def post(self):
        # Call parent class first for default behaviour.
        super(PIMixin, self).post()

        # Write output
        if not self.equidistant:
            # Overrule/write the time range in the export placeholder.
            # Only needed for non-equidistant, because we can't build the
            # times automatically from global start/end datetime.
            self._timeseries_export.times           = self._timeseries_import.times
            self._timeseries_export._start_datetime = self._timeseries_import._start_datetime
            self._timeseries_export._end_datetime   = self._timeseries_import._end_datetime

        self._timeseries_export.resize(
            self._timeseries_import.forecast_datetime, self._timeseries_import.end_datetime)

        times = self.times()
        for ensemble_member in range(self.ensemble_size):
            results = self.extract_results(ensemble_member)

            for variable in self._timeseries_export._values[ensemble_member].keys():
                try:
                    values = results[variable]
                    if len(values) != len(times):
                        values = self.interpolate(
                            times, self.times(variable), values)
                except KeyError:
                    try:
                        ts = self.get_timeseries(variable, ensemble_member)
                        if len(ts.times) != len(times):
                            values = self.interpolate(
                                times, ts.times, ts.values)
                        else:
                            values = ts.values
                    except KeyError:
                        logger.error(
                            "Output requested for non-existent variable {}".format(variable))
                        continue
                self._timeseries_export.set(
                    variable, values, ensemble_member=ensemble_member)

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

    def get_timeseries(self, variable, ensemble_member=0):
        return Timeseries(self._timeseries_import_times, self._timeseries_import.get(variable, ensemble_member=ensemble_member))

    def set_timeseries(self, variable, timeseries, ensemble_member=0, output=True, check_consistency=True):
        if output:
            self._output_timeseries.add(variable)
        if isinstance(timeseries, Timeseries):
            # TODO: add better check on timeseries.times?
            if check_consistency:
                if not np.array_equal(self._timeseries_import_times, timeseries.times):
                    raise Exception("PI: Trying to set/append timeseries {} with different times (in seconds) than the imported timeseries. Please make sure the timeseries covers startDate through endData of the longest imported timeseries.".format(variable))
        else:
            timeseries = Timeseries(self.times(), timeseries)
            assert(len(timeseries.times) == len(timeseries.values))
        self._timeseries_import.set(
            variable, timeseries.values, ensemble_member=ensemble_member)

    def get_forecast_index(self):
        return self._timeseries_import.forecast_index

    def timeseries_at(self, variable, t, ensemble_member=0):
        return self.interpolate(t, self._timeseries_import_times, self._timeseries_import.get(variable, ensemble_member=ensemble_member))

    @property
    def output_variables(self):
        variables = super(PIMixin, self).output_variables
        variables.extend([MX.sym(variable)
                          for variable in self._output_timeseries])
        return variables

    def min_timeseries_id(self, variable):
        """
        Returns the name of the lower bound timeseries for the specified variable.

        :param variable: Variable name
        """
        return '_'.join((variable, 'Min'))

    def max_timeseries_id(self, variable):
        """
        Returns the name of the upper bound timeseries for the specified variable.

        :param variable: Variable name
        """
        return '_'.join((variable, 'Max'))
