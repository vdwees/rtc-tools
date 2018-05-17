import bisect
import logging
from datetime import timedelta

import casadi as ca

import numpy as np

import rtctools.data.pi as pi
import rtctools.data.rtc as rtc
from rtctools._internal.alias_tools import AliasDict
from rtctools._internal.caching import cached

from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


class PIMixin(OptimizationProblem):
    """
    Adds `Delft-FEWS Published Interface
    <https://publicwiki.deltares.nl/display/FEWSDOC/The+Delft-Fews+Published+Interface>`_
    I/O to your optimization problem.

    During preprocessing, files named ``rtcDataConfig.xml``, ``timeseries_import.xml``, ``rtcParameterConfig.xml``,
    and ``rtcParameterConfig_Numerical.xml`` are read from the ``input`` subfolder.  ``rtcDataConfig.xml`` maps
    tuples of FEWS identifiers, including location and parameter ID, to RTC-Tools time series identifiers.

    During postprocessing, a file named ``timeseries_export.xml`` is written to the ``output`` subfolder.

    :cvar pi_binary_timeseries:
        Whether to use PI binary timeseries format. Default is ``False``.
    :cvar pi_parameter_config_basenames:
        List of parameter config file basenames to read. Default is [``rtcParameterConfig``].
    :cvar pi_parameter_config_numerical_basename:
        Numerical config file basename to read. Default is ``rtcParameterConfig_Numerical``.
    :cvar pi_check_for_duplicate_parameters:
        Check if duplicate parameters are read. Default is ``False``.
    :cvar pi_validate_timeseries:
        Check consistency of timeseries. Default is ``True``.
    """

    #: Whether to use PI binary timeseries format
    pi_binary_timeseries = False

    #: Location of rtcParameterConfig files
    pi_parameter_config_basenames = ['rtcParameterConfig']
    pi_parameter_config_numerical_basename = 'rtcParameterConfig_Numerical'

    #: Check consistency of timeseries
    pi_validate_timeseries = True

    #: Check for duplicate parameters
    pi_check_for_duplicate_parameters = True

    # I/O Basenames (can be overridden)
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

        # Additional output variables
        self.__output_timeseries = set()

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
        except IOError:
            raise Exception(
                "PIMixin: {}.xml not found in {}.".format(pi_parameter_config_basename, self.__input_folder))

        try:
            self.__parameter_config_numerical = pi.ParameterConfig(
                  self.__input_folder, self.pi_parameter_config_numerical_basename)
        except IOError:
            self.__parameter_config_numerical = None

        try:
            self.__timeseries_import = pi.Timeseries(
                self.__data_config, self.__input_folder, self.timeseries_import_basename,
                binary=self.pi_binary_timeseries, pi_validate_times=self.pi_validate_timeseries)
        except IOError:
            raise Exception("PIMixin: {}.xml not found in {}.".format(
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
                    raise Exception(
                        'PIMixin: Time stamps must be strictly increasing.')

        if self.equidistant:
            # Check if the timeseries are truly equidistant
            if self.pi_validate_timeseries:
                dt = self.__timeseries_import_times[
                    1] - self.__timeseries_import_times[0]
                for i in range(len(self.__timeseries_import_times) - 1):
                    if self.__timeseries_import_times[i + 1] - self.__timeseries_import_times[i] != dt:
                        raise Exception(
                            'PIMixin: Expecting equidistant timeseries, the time step '
                            'towards {} is not the same as the time step(s) before. Set '
                            'unit to nonequidistant if this is intended.'.format(
                                self.__timeseries_import.times[i + 1]))

        # Stick timeseries into an AliasDict
        self.__timeseries_import_dict = [
            AliasDict(self.alias_relation) for ensemble_member in range(self.ensemble_size)]
        for ensemble_member in range(self.ensemble_size):
            for key, value in self.__timeseries_import.items(ensemble_member):
                self.__timeseries_import_dict[ensemble_member][key] = value

    @cached
    def times(self, variable=None):
        return self.__timeseries_import_times[self.__timeseries_import.forecast_index:]

    @property
    def equidistant(self):
        if self.__timeseries_import.dt is not None:
            return True
        else:
            return False

    def solver_options(self):
        # Call parent
        options = super().solver_options()

        # Only do this if we have a rtcParameterConfig_Numerical
        if self.__parameter_config_numerical is None:
            return options

        # Load solver options from parameter config
        for _location_id, _model, option, value in self.__parameter_config_numerical:
            options[option] = value

        # Done
        return options

    @cached
    def parameters(self, ensemble_member):
        # Call parent class first for default values.
        parameters = super().parameters(ensemble_member)

        # Load parameters from parameter config
        for parameter_config in self.__parameter_config:
            for location_id, model_id, parameter_id, value in parameter_config:
                try:
                    parameter = self.__data_config.parameter(parameter_id, location_id, model_id)
                except KeyError:
                    parameter = parameter_id

                if self.pi_check_for_duplicate_parameters:
                    if parameter in parameters.keys():
                        logger.warning(
                            'PIMixin: parameter {} defined in file {} was already '
                            'present. Using value {}.'.format(
                                parameter, parameter_config.path, value))

                parameters[parameter] = value

        # Done
        return parameters

    @cached
    def constant_inputs(self, ensemble_member):
        # Call parent class first for default values.
        constant_inputs = super().constant_inputs(ensemble_member)

        # Load bounds from timeseries
        for variable in self.dae_variables['constant_inputs']:
            variable = variable.name()
            try:
                timeseries = Timeseries(
                    self.__timeseries_import_times, self.__timeseries_import_dict[ensemble_member][variable])
            except KeyError:
                pass
            else:
                if np.any(np.isnan(timeseries.values[self.__timeseries_import.forecast_index:])):
                    raise Exception("Constant input {} contains NaN".format(variable))
                constant_inputs[variable] = timeseries
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("Read constant input {}".format(variable))
        return constant_inputs

    @cached
    def bounds(self):
        # Call parent class first for default values.
        bounds = super().bounds()

        # Load bounds from timeseries
        for variable in self.dae_variables['free_variables']:
            variable = variable.name()

            m, M = None, None

            timeseries_id = self.min_timeseries_id(variable)
            try:
                m = self.__timeseries_import_dict[0][timeseries_id][
                    self.__timeseries_import.forecast_index:]
            except KeyError:
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("Read lower bound for variable {}".format(variable))

            timeseries_id = self.max_timeseries_id(variable)
            try:
                M = self.__timeseries_import_dict[0][timeseries_id][
                    self.__timeseries_import.forecast_index:]
            except KeyError:
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("Read upper bound for variable {}".format(variable))

            # Replace NaN with +/- inf, and create Timeseries objects
            if m is not None:
                m[np.isnan(m)] = np.finfo(m.dtype).min
                m = Timeseries(self.__timeseries_import_times[
                               self.__timeseries_import.forecast_index:], m)
            if M is not None:
                M[np.isnan(M)] = np.finfo(M.dtype).max
                M = Timeseries(self.__timeseries_import_times[
                               self.__timeseries_import.forecast_index:], M)

            # Store
            if m is not None or M is not None:
                bounds[variable] = (m, M)
        return bounds

    @cached
    def history(self, ensemble_member):
        # Load history
        history = AliasDict(self.alias_relation)

        end_index = self.__timeseries_import.forecast_index + 1
        variable_list = self.dae_variables['states'] + self.dae_variables['algebraics'] + \
            self.dae_variables['control_inputs'] + self.dae_variables['constant_inputs']

        for variable in variable_list:
            variable = variable.name()
            try:
                history[variable] = Timeseries(
                    self.__timeseries_import_times[:end_index],
                    self.__timeseries_import_dict[ensemble_member][variable][:end_index])
            except KeyError:
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("Read history for state {}".format(variable))
        return history

    @property
    def initial_time(self):
        return 0.0

    @cached
    def initial_state(self, ensemble_member):
        history = self.history(ensemble_member)
        return AliasDict(
            self.alias_relation,
            {variable: timeseries.values[-1] for variable, timeseries in history.items()})

    @cached
    def seed(self, ensemble_member):
        # Call parent class first for default values.
        seed = super().seed(ensemble_member)

        # Load seeds
        for variable in self.dae_variables['free_variables']:
            variable = variable.name()
            try:
                s = Timeseries(
                    self.__timeseries_import_times,
                    self.__timeseries_import_dict[ensemble_member][variable])
            except KeyError:
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("PIMixin: Seeded free variable {}".format(variable))
                # A seeding of NaN means no seeding
                s.values[np.isnan(s.values)] = 0.0
                seed[variable] = s
        return seed

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
        self.__timeseries_export.ensemble_size = self.ensemble_size
        self.__timeseries_export.contains_ensemble = self.__timeseries_import.contains_ensemble

        # Start looping over the ensembles for extraction of the output values.
        times = self.times()
        for ensemble_member in range(self.ensemble_size):
            results = self.extract_results(ensemble_member)

            # For all variables that are output variables the values are
            # extracted from the results.
            for variable in [sym.name() for sym in self.output_variables]:
                try:
                    values = results[variable]
                    if len(values) != len(times):
                        values = self.interpolate(
                            times, self.times(variable), values, self.interpolation_method(variable))
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
                            'PIMixin: Output requested for non-existent variable {}. '
                            'Will not be in output file.'.format(variable))
                        continue

                # Check if ID mapping is present
                try:
                    self.__data_config.pi_variable_ids(variable)
                except KeyError:
                    logger.debug(
                        'PIMixin: variable {} has no mapping defined in rtcDataConfig '
                        'so cannot be added to the output file.'.format(variable))
                    continue

                # Add series to output file
                self.__timeseries_export.set(
                    variable, values,
                    unit=self.__timeseries_import.get_unit(variable, ensemble_member=ensemble_member),
                    ensemble_member=ensemble_member)

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

    def get_timeseries(self, variable, ensemble_member=0):
        return Timeseries(self.__timeseries_import_times, self.__timeseries_import_dict[ensemble_member][variable])

    def set_timeseries(self, variable, timeseries, ensemble_member=0, output=True, check_consistency=True, unit=None):

        def stretch_values(values, t_pos):
            # Construct a values range with preceding and possibly following nans
            new_values = np.full_like(self.__timeseries_import_times, np.nan)
            new_values[t_pos:] = values
            return new_values

        if output:
            self.__output_timeseries.add(variable)

        if isinstance(timeseries, Timeseries):
            try:
                assert(len(timeseries.values) == len(timeseries.times))
            except AssertionError:
                raise Exception(
                    'PIMixin: Trying to set timeseries with times and values that are of '
                    'different length (lengths of {} and {}, respectively).'.format(
                        len(timeseries.times), len(timeseries.values)))

            if not np.array_equal(self.__timeseries_import_times, timeseries.times):
                if check_consistency:
                    if not set(self.__timeseries_import_times).issuperset(timeseries.times):
                        raise Exception(
                            'PIMixin: Trying to set/append timeseries {} with different times '
                            '(in seconds) than the imported timeseries. Please make sure '
                            'the timeseries covers startDate through endDate of the longest '
                            'imported timeseries with timestep {}.'.format(
                                variable, self.__timeseries_import.dt))

                # Determine position of first times of added timeseries within the
                # import times. For this we assume that both time ranges are ordered,
                # and that the times of the added series is a subset of the import
                # times.
                t_pos = bisect.bisect_left(self.__timeseries_import_times, timeseries.times[0])

                # Construct a new values range and reconstruct the Timeseries object
                timeseries = Timeseries(self.__timeseries_import_times, stretch_values(timeseries.values, t_pos))

        else:
            if check_consistency:
                try:
                    assert(len(self.times()) == len(timeseries))
                except AssertionError:
                    raise Exception(
                        'PIMixin: Trying to set/append values {} with a different '
                        'length than the forecast length. Please make sure the '
                        'values cover forecastDate through endDate with '
                        'timestep {}.'.format(variable, self.__timeseries_import.dt))

            # If times is not supplied with the timeseries, we add the
            # forecast times range to a new Timeseries object. Hereby
            # we assume that the supplied values stretch from T0 to end.
            t_pos = self.get_forecast_index()

            # Construct a new values range and construct the Timeseries object
            timeseries = Timeseries(self.__timeseries_import_times, stretch_values(timeseries, t_pos))

        if unit is None:
            unit = self.__timeseries_import.get_unit(variable, ensemble_member=ensemble_member)
        self.__timeseries_import.set(
            variable, timeseries.values, ensemble_member=ensemble_member, unit=unit)
        self.__timeseries_import_dict[ensemble_member][variable] = timeseries.values

    def get_forecast_index(self):
        return self.__timeseries_import.forecast_index

    def timeseries_at(self, variable, t, ensemble_member=0):
        return self.interpolate(
            t, self.__timeseries_import_times, self.__timeseries_import_dict[ensemble_member][variable])

    @property
    def ensemble_size(self):
        return self.__timeseries_import.ensemble_size

    @property
    def output_variables(self):
        variables = super().output_variables
        variables.extend([ca.MX.sym(variable)
                          for variable in self.__output_timeseries])
        return variables

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

    def min_timeseries_id(self, variable: str) -> str:
        """
        Returns the name of the lower bound timeseries for the specified variable.

        :param variable: Variable name
        """
        return '_'.join((variable, 'Min'))

    def max_timeseries_id(self, variable: str) -> str:
        """
        Returns the name of the upper bound timeseries for the specified variable.

        :param variable: Variable name
        """
        return '_'.join((variable, 'Max'))
