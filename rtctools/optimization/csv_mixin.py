from datetime import datetime, timedelta
from casadi import MX
import numpy as np
import logging
import os
import sys
import rtctools.data.csv as csv

from .timeseries import Timeseries
from .optimization_problem import OptimizationProblem
from .alias_tools import AliasDict
from .caching import cached

logger = logging.getLogger("rtctools")


class CSVMixin(OptimizationProblem):
    """
    Adds reading and writing of CSV timeseries and parameters to your optimization problem.

    During preprocessing, files named ``timeseries_import.csv``, ``initial_state.csv``, and ``parameters.csv`` are read from the ``input`` subfolder.

    During postprocessing, a file named ``timeseries_export.csv`` is written to the ``output`` subfolder.

    In ensemble mode, a file named ``ensemble.csv`` is read from the ``input`` folder.  This file contains two columns.
    The first column gives the name of the ensemble member, and the second column its probability.  Furthermore, the other XML files
    appear one level deeper inside the filesystem hierarchy, inside subfolders with the names of the ensemble members.

    :cvar csv_delimiter:           Column delimiter used in CSV files.  Default is ``,``.
    :cvar csv_equidistant:         Whether or not the timeseries data is equidistant.  Default is ``True``.
    :cvar csv_ensemble_mode:       Whether or not to use ensembles.  Default is ``False``.
    :cvar csv_validate_timeseries: Check consistency of timeseries.  Default is ``True``.
    """

    #: Column delimiter used in CSV files
    csv_delimiter = ','

    #: Whether or not the timeseries data is equidistant
    csv_equidistant = True

    #: Whether or not to use ensembles
    csv_ensemble_mode = False

    #: Check consistency of timeseries
    csv_validate_timeseries = True

    def __init__(self, **kwargs):
        # Check arguments
        assert('input_folder' in kwargs)
        assert('output_folder' in kwargs)

        # Save arguments
        self._input_folder = kwargs['input_folder']
        self._output_folder = kwargs['output_folder']

        # Additional output variables
        self._output_timeseries = set()

        # Call parent class first for default behaviour.
        super(CSVMixin, self).__init__(**kwargs)

    def pre(self):
        # Call parent class first for default behaviour.
        super(CSVMixin, self).pre()

        # Helper function to check if initiale state array actually defines
        # only the initial state
        def check_initial_state_array(initial_state):
            """
            Check length of initial state array, throw exception when larger than 1.
            """
            if initial_state.shape:
                raise Exception("CSVMixin: Initial state file {} contains more than one row of data. Please remove the data row(s) that do not describe the initial state.".format(
                    os.path.join(self._input_folder, 'initial_state.csv')))

        # Read CSV files
        self._timeseries = []
        self._parameters = []
        self._initial_state = []
        if self.csv_ensemble_mode:
            self._ensemble = np.genfromtxt(os.path.join(
                self._input_folder, 'ensemble.csv'), delimiter=",", deletechars='', dtype=None, names=True, converters={0: str})
            logger.debug("CSVMixin: Read ensemble description")

            for ensemble_member_name in self._ensemble['name']:
                _timeseries = csv.load(os.path.join(self._input_folder, ensemble_member_name,
                                                    'timeseries_import.csv'), delimiter=self.csv_delimiter, with_time=True)
                self._timeseries_times = _timeseries[_timeseries.dtype.names[0]]
                self._timeseries.append(
                    AliasDict(self.alias_relation, {key: np.asarray(_timeseries[key], dtype=np.float64) for key in _timeseries.dtype.names[1:]}))
            logger.debug("CSVMixin: Read timeseries")

            for ensemble_member_name in self._ensemble['name']:
                try:
                    _parameters = csv.load(os.path.join(
                        self._input_folder, ensemble_member_name, 'parameters.csv'), delimiter=self.csv_delimiter)
                    _parameters = {key: float(_parameters[key]) for key in _parameters.dtype.names}
                except IOError:
                    _parameters = {}
                self._parameters.append(AliasDict(self.alias_relation, _parameters))
            logger.debug("CSVMixin: Read parameters.")

            for ensemble_member_name in self._ensemble['name']:
                try:
                    _initial_state = csv.load(os.path.join(
                        self._input_folder, ensemble_member_name, 'initial_state.csv'), delimiter=self.csv_delimiter)
                    check_initial_state_array(_initial_state)
                    _initial_state = {key: float(_initial_state[key]) for key in _initial_state.dtype.names}
                except IOError:
                    _initial_state = {}
                self._initial_state.append(AliasDict(self.alias_relation, _initial_state))
            logger.debug("CSVMixin: Read initial state.")
        else:
            _timeseries = csv.load(os.path.join(
                self._input_folder, 'timeseries_import.csv'), delimiter=self.csv_delimiter, with_time=True)
            self._timeseries_times = _timeseries[_timeseries.dtype.names[0]]
            self._timeseries.append(
                AliasDict(self.alias_relation, {key: np.asarray(_timeseries[key], dtype=np.float64) for key in _timeseries.dtype.names[1:]}))
            logger.debug("CSVMixin: Read timeseries.")

            try:
                _parameters = csv.load(os.path.join(
                    self._input_folder, 'parameters.csv'), delimiter=self.csv_delimiter)
                logger.debug("CSVMixin: Read parameters.")
                _parameters = {key: float(_parameters[key]) for key in _parameters.dtype.names}
            except IOError:
                _parameters = {}
            self._parameters.append(AliasDict(self.alias_relation, _parameters))

            try:
                _initial_state = csv.load(os.path.join(
                    self._input_folder, 'initial_state.csv'), delimiter=self.csv_delimiter)
                logger.debug("CSVMixin: Read initial state.")
                check_initial_state_array(_initial_state)
                _initial_state = {key: float(_initial_state[key]) for key in _initial_state.dtype.names}
            except IOError:
                _initial_state = {}
            self._initial_state.append(AliasDict(self.alias_relation, _initial_state))

        self._timeseries_times_sec = self._datetime_to_sec(
            self._timeseries_times)

        # Timestamp check
        if self.csv_validate_timeseries:
            for i in range(len(self._timeseries_times_sec) - 1):
                if self._timeseries_times_sec[i] >= self._timeseries_times_sec[i + 1]:
                    raise Exception(
                        'CSVMixin: Time stamps must be strictly increasing.')

        if self.csv_equidistant:
            # Check if the timeseries are truly equidistant
            if self.csv_validate_timeseries:
                dt = self._timeseries_times_sec[
                    1] - self._timeseries_times_sec[0]
                for i in range(len(self._timeseries_times_sec) - 1):
                    if self._timeseries_times_sec[i + 1] - self._timeseries_times_sec[i] != dt:
                        raise Exception('CSVMixin: Expecting equidistant timeseries, the time step towards {} is not the same as the time step(s) before. Set equidistant=False if this is intended.'.format(
                            self._timeseries_times[i + 1]))

    def times(self, variable=None):
        return self._timeseries_times_sec

    @property
    def equidistant(self):
        return self.csv_equidistant

    @property
    def ensemble_size(self):
        if self.csv_ensemble_mode:
            return len(self._ensemble['probability'])
        else:
            return 1

    def ensemble_member_probability(self, ensemble_member):
        if self.csv_ensemble_mode:
            return self._ensemble['probability'][ensemble_member]
        else:
            return 1.0

    @cached
    def parameters(self, ensemble_member):
        # Call parent class first for default values.
        parameters = super(CSVMixin, self).parameters(ensemble_member)

        # Load parameters from parameter config
        for parameter in self.dae_variables['parameters']:
            parameter = parameter.name()
            try:
                parameters[parameter] = self._parameters[ensemble_member][parameter]
            except KeyError:
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("CSVMixin: Read parameter {} ".format(parameter))
        return parameters

    @cached
    def constant_inputs(self, ensemble_member):
        # Call parent class first for default values.
        constant_inputs = super(
            CSVMixin, self).constant_inputs(ensemble_member)

        # Load bounds from timeseries
        for variable in self.dae_variables['constant_inputs']:
            variable = variable.name()
            try:
                constant_inputs[variable] = Timeseries(
                    self._timeseries_times_sec, self._timeseries[ensemble_member][variable])
            except (KeyError, ValueError):
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("CSVMixin: Read constant input {}".format(variable))

        return constant_inputs

    @cached
    def bounds(self):
        # Call parent class first for default values.
        bounds = super(CSVMixin, self).bounds()

        # Load bounds from timeseries
        for variable in self.dae_variables['free_variables']:
            variable = variable.name()

            m, M = None, None

            timeseries_id = self.min_timeseries_id(variable)
            try:
                m = self._timeseries[0][timeseries_id]
            except (KeyError, ValueError):
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("CSVMixin: Read lower bound for variable {}".format(variable))

            timeseries_id = self.max_timeseries_id(variable)
            try:
                M = self._timeseries[0][timeseries_id]
            except (KeyError, ValueError):
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("CSVMixin: Read upper bound for variable {}".format(variable))

            # Replace NaN with +/- inf, and create Timeseries objects
            if m != None:
                m[np.isnan(m)] = np.finfo(m.dtype).min
                m = Timeseries(self._timeseries_times_sec, m)
            if M != None:
                M[np.isnan(M)] = np.finfo(M.dtype).max
                M = Timeseries(self._timeseries_times_sec, M)

            # Store
            if m != None or M != None:
                bounds[variable] = (m, M)
        return bounds

    @property
    def initial_time(self):
        return 0.0

    @cached
    def initial_state(self, ensemble_member):
        # Call parent class first for default values.
        initial_state = super(CSVMixin, self).initial_state(ensemble_member)

        # Load parameters from parameter config
        for variable in self.dae_variables['free_variables']:
            variable = variable.name()
            try:
                initial_state[variable] = self._initial_state[ensemble_member][variable]
            except (KeyError, ValueError):
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("CSVMixin: Read initial state {}".format(variable))
        return initial_state

    @cached
    def seed(self, ensemble_member):
        # Call parent class first for default values.
        seed = super(CSVMixin, self).seed(ensemble_member)

        # Load seed values from CSV
        for variable in self.dae_variables['free_variables']:
            variable = variable.name()
            try:
                s = Timeseries(self._timeseries_times_sec, self._timeseries[ensemble_member][variable])
            except (KeyError, ValueError):
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("CSVMixin: Seeded free variable {}".format(variable))
                # A seeding of NaN means no seeding
                s.values[np.isnan(s.values)] = 0.0
                seed[variable] = s
        return seed

    def post(self):
        # Call parent class first for default behaviour.
        super(CSVMixin, self).post()

        # Write output
        times = self.times()

        def write_output(ensemble_member, folder):
            results = self.extract_results(ensemble_member)
            names = ['time'] + sorted(set([sym.name() for sym in self.output_variables]))
            formats = ['O'] + (len(names) - 1) * ['f8']
            dtype = dict(names=names, formats=formats)
            data = np.zeros(len(self._timeseries_times), dtype=dtype)
            data['time'] = self._timeseries_times
            for i, output_variable in enumerate(self.output_variables):
                output_variable = output_variable.name()
                try:
                    values = results[output_variable]
                    if len(values) != len(times):
                        values = self.interpolate(
                            times, self.times(output_variable), values, self.interpolation_method(output_variable))
                except KeyError:
                    try:
                        ts = self.get_timeseries(
                            output_variable, ensemble_member)
                        if len(ts.times) != len(times):
                            values = self.interpolate(
                                times, ts.times, ts.values)
                        else:
                            values = ts.values
                    except KeyError:
                        logger.error(
                            "Output requested for non-existent variable {}".format(output_variable))
                        continue
                data[output_variable] = values

            fname = os.path.join(folder, 'timeseries_export.csv')
            csv.save(fname, data, delimiter=self.csv_delimiter, with_time=True)

        if self.csv_ensemble_mode:
            for ensemble_member, ensemble_member_name in enumerate(self._ensemble['name']):
                write_output(ensemble_member, os.path.join(
                    self._output_folder, ensemble_member_name))
        else:
            write_output(0, self._output_folder)

    def _datetime_to_sec(self, d):
        # Return the date/timestamps in seconds since t0.
        if hasattr(d, '__iter__'):
            return np.array([(t - self._timeseries_times[0]).total_seconds() for t in d])
        else:
            return (d - self._timeseries_times[0]).total_seconds()

    def _sec_to_datetime(self, s):
        # Return the date/timestamps in seconds since t0 as datetime objects.
        if hasattr(s, '__iter__'):
            return [self._timeseries_times[0] + timedelta(seconds=t) for t in s]
        else:
            return self._timeseries_times[0] + timedelta(seconds=s)

    def get_timeseries(self, variable, ensemble_member=0):
        return Timeseries(self._timeseries_times_sec, self._timeseries[ensemble_member][variable])

    def set_timeseries(self, variable, timeseries, ensemble_member=0, output=True, check_consistency=True):
        if output:
            self._output_timeseries.add(variable)
        if isinstance(timeseries, Timeseries):
            # TODO: add better check on timeseries.times?
            if check_consistency:
                if not np.array_equal(self.times(), timeseries.times):
                    raise Exception("CSV: Trying to set/append timeseries {} with different times (in seconds) than the imported timeseries. Please make sure the timeseries covers startDate through endData of the longest imported timeseries.".format(variable))
        else:
            timeseries = Timeseries(self.times(), timeseries)
            assert(len(timeseries.times) == len(timeseries.values))
        self._timeseries[ensemble_member][variable] = timeseries.values

    def timeseries_at(self, variable, t, ensemble_member=0):
        return self.interpolate(t, self._timeseries_times_sec, self._timeseries[ensemble_member][variable])

    @property
    def output_variables(self):
        variables = super(CSVMixin, self).output_variables
        variables.extend([MX.sym(variable)
                          for variable in self._output_timeseries])
        return variables

    def min_timeseries_id(self, variable):
        """
        Returns the name of the lower bound timeseries for the specified variable.

        :param variable: Variable name.
        """
        return '_'.join((variable, 'Min'))

    def max_timeseries_id(self, variable):
        """
        Returns the name of the upper bound timeseries for the specified variable.

        :param variable: Variable name.
        """
        return '_'.join((variable, 'Max'))
