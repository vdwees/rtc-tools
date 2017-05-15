# cython: embedsignature=True

from casadi import Function, nlpsol, MX, CasadiOptions, vertcat
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import itertools
import logging
#import cython

from timeseries import Timeseries
from alias_tools import AliasDict

logger = logging.getLogger("rtctools")


class Alias(object):
    """
    Variable alias.

    .. deprecated:: 2.0.0-beta6
    """

    def __init__(self, name, negated):
        """
        Create a new alias object.

        :param name:    Alias name.
        :param negated: ``True`` if the alias has the opposite sign of the original variable.
        """
        self._name = name
        self._negated = negated

    @property
    def name(self):
        """
        The name of the alias.
        """
        return self._name

    @property
    def sign(self):
        """
        The sign of the alias, such that the alias variable is sign times the original variable.
        """
        if self._negated:
            return -1
        else:
            return 1


class LookupTable(object):
    """
    Lookup table.
    """

    def __init__(self, inputs, function):
        """
        Create a new lookup table object.

        :param inputs: List of lookup table input variables.
        :param function: Lookup table CasADi :class:`Function`.
        """
        self._inputs = inputs
        self._function = function

    @property
    def inputs(self):
        """
        List of lookup table input variables.
        """
        return self._inputs

    @property
    def function(self):
        """
        Lookup table CasADi :class:`Function`.
        """
        return self._function

    def __call__(self, *args):
        """
        Evaluate the lookup table.

        :param args: Input values.
        :type args: Float, iterable of floats, or :class:`Timeseries`
        :returns: Lookup table evaluated at input values.

        Example use::

            y = lookup_table(1.0)
            [y1, y2] = lookup_table([1.0, 2.0])

        """
        if isinstance(args[0], Timeseries):
            return Timeseries(args[0].times, self(args[0].values))
        else:
            if hasattr(args[0], '__iter__'):
                evaluator = np.vectorize(
                    lambda v: float(self.function([v])[0]))
                return evaluator(args[0])
            else:
                return float(self.function(args)[0])


class OptimizationProblem(object):
    """
    Base class for all optimization problems.
    """

    __metaclass__ = ABCMeta

    def optimize(self, preprocessing=True, postprocessing=True):
        """
        Perform one initialize-transcribe-solve-finalize cycle.

        :param preprocessing:  True to enable a call to ``pre`` preceding the opimization.
        :param postprocessing: True to enable a call to ``post`` following the optimization.

        :returns: True on success.
        """

        logger.info("Entering optimize()")

        # Do any preprocessing, which may include changing parameter values on
        # the model
        if preprocessing:
            self.pre()

            # Check if control inputs are bounded
            self._check_bounds_control_input()
        else:
            logger.debug(
                'Skipping Preprocessing in OptimizationProblem.optimize()')

        # Transcribe problem
        discrete, lbx, ubx, lbg, ubg, x0, nlp = self.transcribe()

        # Create an NLP solver
        logger.debug("Collecting solver options")

        self._mixed_integer = np.any(discrete)
        options = {}
        options.update(self.solver_options()) # Create a copy
        if self._mixed_integer:
            options['discrete'] = discrete

        logger.debug("Creating solver")

        my_solver = options['solver']
        del options['solver']

        CasadiOptions.setOptimizedNumDir(options['optimized_num_dir'])
        del options['optimized_num_dir']

        solver = nlpsol('nlp', my_solver, nlp, options)

        # Solve NLP
        logger.info("Calling solver")

        results = solver(x0 = x0, lbx = lbx, ubx = ubx, lbg = vertcat(*lbg), ubg = vertcat(*ubg))

        # Extract relevant stats
        self._objective_value = float(results['f'])
        self._solver_output = results['x']
        self._solver_stats = solver.getStats()

        # Get the return status
        if self._solver_stats['return_status'] in ['Solve_Succeeded', 'Solved_To_Acceptable_Level', 'User_Requested_Stop', 'SUCCESS']:
            logger.info("Solver succeeded with status {}".format(
                self._solver_stats['return_status']))

            success = True
        elif self._solver_stats['return_status'] in ['Not_Enough_Degrees_Of_Freedom']:
            logger.warning("Solver failed with status {}".format(
                self._solver_stats['return_status']))

            success = False
        else:
            logger.error("Solver failed with status {}".format(
                self._solver_stats['return_status']))

            success = False

        # Do any postprocessing
        if postprocessing:
            self.post()
        else:
            logger.debug(
                'Skipping Postprocessing in OptimizationProblem.optimize()')

        # Done
        logger.info("Done with optimize()")

        return success

    def _check_bounds_control_input(self):
        # Checks if at the control inputs have bounds, log warning when a control input is not bounded.
        bounds = self.bounds()

        for variable in self.dae_variables['control_inputs']:
            variable = variable.getName()
            if variable not in bounds:
                logger.warning(
                    "OptimizationProblem: control input {} has no bounds.".format(variable))                

    @abstractmethod
    def transcribe(self):
        """
        Transcribe the continuous optimization problem to a discretized, solver-ready
        optimization problem.
        """
        pass

    def solver_options(self):
        """
        Returns a dictionary of CasADi optimization problem solver options.

        The default solver for continuous problems is `Ipopt <https://projects.coin-or.org/Ipopt/>`_.  The default solver for mixed integer problems is `Bonmin <http://projects.coin-or.org/Bonmin/>`_.

        :returns: A dictionary of CasADi :class:`NlpSolver` options.  See the CasADi, Ipopt, and Bonmin documentation for details.
        """
        options = {'optimized_num_dir': 3, 'inputs_check': False, 'ignore_check_vec': True, 'regularity_check': False}
        if self._mixed_integer:
            options['solver'] = 'bonmin'
            options['algorithm'] = 'B-BB'
            options['nlp_solver'] = 'Ipopt'
            options['nlp_log_level'] = 2
            options['linear_solver'] = 'mumps'
            options['print_level'] = 5
        else:
            options['solver'] = 'ipopt'
            options['linear_solver'] = 'mumps'
            options['print_level'] = 5
        return options

    @abstractproperty
    def solver_input(self):
        """
        The symbolic input to the NLP solver.
        """
        pass

    @abstractmethod
    def extract_results(self, ensemble_member=0):
        """
        Extracts state and control input time series from optimizer results.

        :returns: A dictionary of result time series.
        """
        pass

    @property
    def objective_value(self):
        """
        The last obtained objective function value.
        """
        return self._objective_value

    @property
    def solver_output(self):
        """
        The raw output from the last NLP solver run.
        """
        return self._solver_output

    @property
    def solver_stats(self):
        """
        The stats from the last NLP solver run.
        """
        return self._solver_stats

    def pre(self):
        """
        Preprocessing logic is performed here.
        """
        pass

    @abstractproperty
    def dae_residual(self):
        """
        Symbolic DAE residual of the model.
        """
        pass

    @abstractproperty
    def dae_variables(self):
        """
        Dictionary of symbolic variables for the DAE residual.
        """
        pass

    @property
    def path_variables(self):
        """
        List of additional, time-dependent optimization variables, not covered by the DAE model.
        """
        return []

    @property
    def extra_variables(self):
        """
        List of additional, time-independent optimization variables, not covered by the DAE model.
        """
        return []

    @property
    def output_variables(self):
        """
        List of variables that the user requests to be included in the output files.
        """
        return []

    def delayed_feedback(self):
        """
        Returns the delayed feedback mappings.  These are given as a list of triples :math:`(x, y, \\tau)`,
        to indicate that :math:`y = x(t - \\tau)`.

        :returns: A list of triples.

        Example::

            def delayed_feedback(self):
                fb1 = ['x', 'y', 0.1]
                fb2 = ['x', 'z', 0.2]
                return [fb1, fb2]

        """
        return []

    @property
    def ensemble_size(self):
        """
        The number of ensemble members.
        """
        return 1

    def ensemble_member_probability(self, ensemble_member):
        """
        The probability of an ensemble member occurring.

        :param ensemble_member: The ensemble member index.

        :returns: The probability of an ensemble member occurring.

        :raises: IndexError
        """
        return 1.0

    def parameters(self, ensemble_member):
        """
        Returns a dictionary of parameters.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of parameter names and values.
        """
        return AliasDict(self.alias_relation)

    def dynamic_parameters(self):
        """
        Returns a list of parameter symbols that may vary from run to run.  The values
        of these parameters are not cached.

        :returns: A list of parameter symbols.
        """
        return []

    def constant_inputs(self, ensemble_member):
        """
        Returns a dictionary of constant inputs.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of constant input names and time series.
        """
        return AliasDict(self.alias_relation)

    def lookup_tables(self, ensemble_member):
        """
        Returns a dictionary of lookup tables.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of variable names and lookup tables.
        """
        return AliasDict(self.alias_relation)

    def bounds(self):
        """
        Returns variable bounds as a dictionary mapping variable names to a pair of bounds.
        A bound may be a constant, or a time series.

        :returns: A dictionary of variable names and ``(upper, lower)`` bound pairs. The bounds may be numbers or :class:`Timeseries` objects.

        Example::

            def bounds(self):
                return {'x': (1.0, 2.0), 'y': (2.0, 3.0)}

        """
        return AliasDict(self.alias_relation)

    def history(self, ensemble_member):
        """
        Returns the state history.  Uses the initial_state() method by default.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of variable names and historical time series (up to and including t0).
        """
        initial_state = self.initial_state(ensemble_member)
        return AliasDict(self.alias_relation, {variable: Timeseries(np.array([self.initial_time]), np.array([state])) for variable, state in initial_state.items()})

    @abstractproperty
    def alias_relation(self):
        raise NotImplementedError

    def variable_is_discrete(self, variable):
        """
        Returns ``True`` if the provided variable is discrete.

        :param variable: Variable name.

        :returns: ``True`` if variable is discrete (integer).
        """
        return False

    def variable_aliases(self, variable):
        """
        Returns a list of aliases for the given variable.

        .. deprecated:: 2.0.0-beta6
           Use :func:`alias_relation.aliases` instead.

        :param variable: Variable name.

        :returns: A list of variable aliases.
        """
        return [Alias(variable, False)]

    def variable_nominal(self, variable):
        """
        Returns the nominal value of the variable.  Variables are scaled by replacing them with
        their nominal value multiplied by the new variable.

        :param variable: Variable name.

        :returns: The nominal value of the variable.
        """
        return 1

    @property
    def initial_time(self):
        """
        The initial time in seconds.
        """
        return self.times()[0]

    def initial_state(self, ensemble_member):
        """
        The initial state.

        The default implementation uses t0 data returned by the ``history`` method.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of variable names and initial state (t0) values.
        """
        t0 = self.initial_time
        history = self.history(ensemble_member)
        return {variable: self.interpolate(t0, timeseries.times, timeseries.values) for variable, timeseries in history.items()}

    @property
    def initial_residual(self):
        """
        The initial equation residual.

        Initial equations are used to find consistent initial conditions.

        :returns: An :class:`MX` object representing F in the initial equation F = 0.
        """
        return MX(0)

    def seed(self, ensemble_member):
        """
        Seeding data.  The optimization algorithm is seeded with the data returned by this method.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of variable names and seed time series.
        """
        return AliasDict(self.alias_relation)

    def objective(self, ensemble_member):
        """
        The objective function for the given ensemble member.

        Call :func:`OptimizationProblem.state_at` to return a symbol representing a model variable at a given time.

        :param ensemble_member: The ensemble member index.

        :returns: An :class:`MX` object representing the objective function.

        Example::

            def objective(self, ensemble_member):
                # Return value of state 'x' at final time:
                times = self.times()
                return self.state_at('x', times[-1], ensemble_member)

        """
        return MX(0)

    def path_objective(self, ensemble_member):
        """
        Returns a path objective the given ensemble member.  Path objectives apply to all times and ensemble members simultaneously.

        Call :func:`OptimizationProblem.state` to return a time- and ensemble-member-independent symbol representing a model variable.

        :param ensemble_member: The ensemble member index.  This index is currently unused, and here for future use only.

        :returns: A :class:`MX` object representing the path objective.

        Example::

            def path_objective(self, ensemble_member):
                # Minimize x(t) for all t
                return self.state('x')

        """
        return MX(0)

    def constraints(self, ensemble_member):
        """
        Returns a list of constraints for the given ensemble member.

        Call :func:`OptimizationProblem.state_at` to return a symbol representing a model variable at a given time.

        :param ensemble_member: The ensemble member index.

        :returns: A list of triples ``(f, m, M)``, with an :class:`MX` object representing the constraint function ``f``, lower bound ``m``, and upper bound ``M``. The bounds must be numbers.

        Example::

            def constraints(self, ensemble_member):
                t = 1.0
                constraint1 = (2 * self.state_at('x', t, ensemble_member), 2.0, 4.0)
                constraint2 = (self.state_at('x', t, ensemble_member) + self.state_at('y', t, ensemble_member), 2.0, 3.0)
                return [constraint1, constraint2]

        """
        return []

    def path_constraints(self, ensemble_member):
        """
        Returns a list of path constraints.  Path constraints apply to all times and ensemble members simultaneously.

        Call :func:`OptimizationProblem.state` to return a time- and ensemble-member-independent symbol representing a model variable.

        :param ensemble_member: The ensemble member index.  This index may only be used to supply member-dependent bounds.

        :returns: A list of triples ``(f, m, M)``, with an :class:`MX` object representing the path constraint function ``f``, lower bound ``m``, and upper bound ``M``.  The bounds may be numbers or :class:`Timeseries` objects.

        Example::

            def path_constraints(self, ensemble_member):
                # 2 * x must lie between 2 and 4 for every time instance.
                path_constraint1 = (2 * self.state('x'), 2.0, 4.0)
                # x + y must lie between 2 and 3 for every time instance
                path_constraint2 = (self.state('x') + self.state('y'), 2.0, 3.0)
                return [path_constraint1, path_constraint2]

        """
        return []

    def post(self):
        """
        Postprocessing logic is performed here.
        """
        pass

    @property
    def equidistant(self):
        """
        ``True`` if all time series are equidistant.
        """
        return False

    #@cython.boundscheck(False)
    def interpolate(self, t, np.ndarray ts, fs, f_left=np.nan, f_right=np.nan):
        """
        Linear interpolation over time.

        :param t:       Time at which to evaluate the interpolant.
        :type t:        float or vector of floats
        :param ts:      Time stamps.
        :type ts:       numpy array
        :param fs:      Function values at time stamps ts.
        :param f_left:  Function value left of leftmost time stamp.
        :param f_right: Function value right of rightmost time stamp.

        :returns: The interpolated value.
        """
        if hasattr(t, '__iter__'):
            f = np.vectorize(lambda t_: self.interpolate(
                t_, ts, fs, f_left, f_right))
            return f(t)

        if t < ts[0]:
            if f_left != None:
                return f_left
            else:
                raise Exception("CSVMixin: Point {} left of range".format(t))
        if t > ts[-1]:
            if f_right != None:
                return f_right
            else:
                raise Exception("CSVMixin: Point {} right of range".format(t))

        if isinstance(ts, np.ndarray):
            n = len(ts)
        else:
            n = ts.size1()

        if self.equidistant:
            if n > 1:
                # We don't cache this, as the user may specify a coarser set of
                # time stamps for optimization variables.
                dt = ts[1] - ts[0]

                (k, r) = divmod(t - ts[0], dt)
                k = int(k)

                if r != 0:
                    return fs[k] + r * (fs[k + 1] - fs[k]) / dt
                else:
                    return fs[k]
            else:
                return fs[0]
        else:
            for i in range(n - 1):
                if t >= ts[i]:
                    if t <= ts[i]:
                        # This special case is needed to avoid interpolation if
                        # not absolutely necessary.  Interpolation is
                        # problematic if one of the interpolants in NaN.
                        return fs[i]
                    elif t < ts[i + 1]:
                        return fs[i] + (fs[i + 1] - fs[i]) / (ts[i + 1] - ts[i]) * (t - ts[i])
            if t == ts[-1]:
                return fs[-1]

    @abstractproperty
    def controls(self):
        """
        List of names of the control variables (excluding aliases).
        """
        pass

    @abstractmethod
    def discretize_controls(self, resolved_bounds):
        """
        Performs the discretization of the control inputs, filling lower and upper
        bound vectors for the resulting optimization variables, as well as an initial guess.

        :param resolved_bounds: :class:`AliasDict` of numerical bound values.  This is the same dictionary as returned by :func:`bounds`,
        but with all parameter symbols replaced with their numerical values.

        :returns: The number of control variables in the optimization problem, a lower bound vector, an upper bound vector, and a seed vector.
        """
        pass

    @abstractmethod
    def extract_controls(self, ensemble_member=0):
        """
        Extracts state time series from optimizer results.

        Must return a dictionary of result time series.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of control input time series.
        """
        pass

    @abstractmethod
    def control_vector(self, variable, ensemble_member=0):
        """
        Return the optimization variables for the entire time horizon of the given state.

        :param variable:        Variable name.
        :param ensemble_member: The ensemble member index.

        :returns: A vector of control input symbols for the entire time horizon.

        :raises: KeyError
        """
        pass

    def control(self, variable):
        """
        Returns an :class:`MX` symbol for the given control input, not bound to any time.

        :param variable: Variable name.

        :returns: :class:`MX` symbol for given control input.

        :raises: KeyError
        """
        return self.variable(variable)

    @abstractmethod
    def control_at(self, variable, t, ensemble_member=0, scaled=False):
        """
        Returns an :class:`MX` symbol representing the given control input at the given time.

        :param variable:        Variable name.
        :param t:               Time.
        :param ensemble_member: The ensemble member index.
        :param scaled:          True to return the scaled variable.

        :returns: :class:`MX` symbol representing the control input at the given time.

        :raises: KeyError
        """
        pass

    @abstractproperty
    def differentiated_states(self):
        """
        List of names of the differentiated state variables (excluding aliases).
        """
        pass

    @abstractproperty
    def algebraic_states(self):
        """
        List of names of the algebraic state variables (excluding aliases).
        """
        pass

    @abstractmethod
    def discretize_states(self, resolved_bounds):
        """
        Perform the discretization of the states, filling lower and upper
        bound vectors for the resulting optimization variables, as well as an initial guess.

        :param resolved_bounds: :class:`AliasDict` of numerical bound values.  This is the same dictionary as returned by :func:`bounds`,
        but with all parameter symbols replaced with their numerical values.

        :returns: The number of control variables in the optimization problem, a lower bound vector, an upper bound vector, and a seed vector.
        """
        pass

    @abstractmethod
    def extract_states(self, ensemble_member=0):
        """
        Extracts state time series from optimizer results.

        Must return a dictionary of result time series.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of state time series.
        """
        pass

    @abstractmethod
    def state_vector(self, variable, ensemble_member=0):
        """
        Return the optimization variables for the entire time horizon of the given state.

        :param variable:        Variable name.
        :param ensemble_member: The ensemble member index.

        :returns: A vector of state symbols for the entire time horizon.

        :raises: KeyError
        """
        pass

    def state(self, variable):
        """
        Returns an :class:`MX` symbol for the given state, not bound to any time.

        :param variable: Variable name.

        :returns: :class:`MX` symbol for given state.

        :raises: KeyError
        """
        return self.variable(variable)

    @abstractmethod
    def state_at(self, variable, t, ensemble_member=0, scaled=False):
        """
        Returns an :class:`MX` symbol representing the given variable at the given time.

        :param variable:        Variable name.
        :param t:               Time.
        :param ensemble_member: The ensemble member index.
        :param scaled:          True to return the scaled variable.

        :returns: :class:`MX` symbol representing the state at the given time.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def extra_variable(self, variable, ensemble_member=0):
        """
        Returns an :class:`MX` symbol representing the extra variable inside the state vector.

        :param variable:        Variable name.
        :param ensemble_member: The ensemble member index.

        :returns: :class:`MX` symbol representing the extra variable.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def states_in(self, variable, t0=None, tf=None, ensemble_member=0):
        """
        Iterates over symbols for states in the interval [t0, tf].

        :param variable:        Variable name.
        :param t0:              Left bound of interval.  If equal to None, the initial time is used.
        :param tf:              Right bound of interval.  If equal to None, the final time is used.
        :param ensemble_member: The ensemble member index.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def integral(self, variable, t0=None, tf=None, ensemble_member=0):
        """
        Returns an expression for the integral over the interval [t0, tf].

        :param variable:        Variable name.
        :param t0:              Left bound of interval.  If equal to None, the initial time is used.
        :param tf:              Right bound of interval.  If equal to None, the final time is used.
        :param ensemble_member: The ensemble member index.

        :returns: :class:`MX` object representing the integral.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def der(self, variable):
        """
        Returns an :class:`MX` symbol for the time derivative given state, not bound to any time.

        :param variable: Variable name.

        :returns: :class:`MX` symbol for given state.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def der_at(self, variable, t, ensemble_member=0):
        """
        Returns an expression for the time derivative of the specified variable at time t.

        :param variable:        Variable name.
        :param t:               Time.
        :param ensemble_member: The ensemble member index.

        :returns: :class:`MX` object representing the derivative.

        :raises: KeyError
        """
        pass

    def get_timeseries(self, variable, ensemble_member=0):
        """
        Looks up a timeseries from the internal data store.

        :param variable:        Variable name.
        :param ensemble_member: The ensemble member index.

        :returns: The requested time series.
        :rtype: :class:`Timeseries`

        :raises: KeyError
        """
        raise NotImplementedError

    def set_timeseries(self, variable, timeseries, ensemble_member=0, output=True, check_consistency=True):
        """
        Sets a timeseries in the internal data store.

        :param variable:          Variable name.
        :param timeseries:        Time series data.
        :type timeseries:         iterable of floats, or :class:`Timeseries`
        :param ensemble_member:   The ensemble member index.
        :param output:            Whether to include this time series in output data files.
        :param check_consistency: Whether to check consistency between the time stamps on the new timeseries object and any existing time stamps.
        """
        raise NotImplementedError

    def timeseries_at(self, variable, t, ensemble_member=0):
        """
        Return the value of a time series at the given time.

        :param variable:        Variable name.
        :param t:               Time.
        :param ensemble_member: The ensemble member index.

        :returns: The interpolated value of the time series.

        :raises: KeyError
        """
        raise NotImplementedError

    def map_path_expression(self, expr, ensemble_member):
        """
        Maps the path expression `expr` over the entire time horizon of the optimization problem.

        :param expr: An :class:`MX` path expression.

        :returns: An :class:`MX` expression evaluating `expr` over the entire time horizon.
        """
        raise NotImplementedError
