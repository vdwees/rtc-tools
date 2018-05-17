import copy
import itertools
import logging
from collections import OrderedDict

import casadi as ca

import numpy as np

import pkg_resources

import pymoca
import pymoca.backends.casadi.api


from rtctools._internal.alias_tools import AliasDict, AliasRelation
from rtctools._internal.caching import cached

logger = logging.getLogger("rtctools")


class SimulationProblem:
    """
    Implements the `BMI <http://csdms.colorado.edu/wiki/BMI_Description>`_ Interface.

    Base class for all Simulation problems. Loads the Modelica Model.

    :cvar modelica_library_folders: Folders containing any referenced Modelica libraries. Default is an empty list.

    """

    # Folders in which the referenced Modelica libraries are found
    modelica_library_folders = []

    def __init__(self, **kwargs):
        # Check arguments
        assert('model_folder' in kwargs)

        # Log pymoca version
        logger.debug("Using pymoca {}.".format(pymoca.__version__))

        # Transfer model from the Modelica .mo file to CasADi using pymoca
        if 'model_name' in kwargs:
            model_name = kwargs['model_name']
        else:
            if hasattr(self, 'model_name'):
                model_name = self.model_name
            else:
                model_name = self.__class__.__name__

        # Load model from pymoca backend
        self.__pymoca_model = pymoca.backends.casadi.api.transfer_model(
            kwargs['model_folder'], model_name, self.compiler_options())

        # Extract the CasADi MX variables used in the model
        self.__mx = {}
        self.__mx['time'] = [self.__pymoca_model.time]
        self.__mx['states'] = [v.symbol for v in self.__pymoca_model.states]
        self.__mx['derivatives'] = [v.symbol for v in self.__pymoca_model.der_states]
        self.__mx['algebraics'] = [v.symbol for v in self.__pymoca_model.alg_states]
        self.__mx['parameters'] = [v.symbol for v in self.__pymoca_model.parameters]
        self.__mx['constant_inputs'] = []
        self.__mx['lookup_tables'] = []

        # TODO: implement delayed feedback
        delayed_feedback_variables = []

        for v in self.__pymoca_model.inputs:
            if v.symbol.name() in delayed_feedback_variables:
                # Delayed feedback variables are local to each ensemble, and
                # therefore belong to the collection of algebraic variables,
                # rather than to the control inputs.
                self.__mx['algebraics'].append(v.symbol)
            else:
                if v.symbol.name() in kwargs.get('lookup_tables', []):
                    self.__mx['lookup_tables'].append(v.symbol)
                else:
                    # All inputs are constant inputs
                    self.__mx['constant_inputs'].append(v.symbol)

        # Log variables in debug mode
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug("SimulationProblem: Found states {}".format(
                ', '.join([var.name() for var in self.__mx['states']])))
            logger.debug("SimulationProblem: Found derivatives {}".format(
                ', '.join([var.name() for var in self.__mx['derivatives']])))
            logger.debug("SimulationProblem: Found algebraics {}".format(
                ', '.join([var.name() for var in self.__mx['algebraics']])))
            logger.debug("SimulationProblem: Found constant inputs {}".format(
                ', '.join([var.name() for var in self.__mx['constant_inputs']])))
            logger.debug("SimulationProblem: Found parameters {}".format(
                ', '.join([var.name() for var in self.__mx['parameters']])))

        # Initialize an AliasDict for nominals and types
        self.__nominals = AliasDict(self.alias_relation)
        self.__python_types = AliasDict(self.alias_relation)
        for v in itertools.chain(
                self.__pymoca_model.states, self.__pymoca_model.alg_states, self.__pymoca_model.inputs):
            sym_name = v.symbol.name()

            # Store the types in an AliasDict
            self.__python_types[sym_name] = v.python_type

            # If the nominal is 0.0 or 1.0 or -1.0, ignore: get_variable_nominal returns a default of 1.0
            # TODO: handle nominal vectors (update() will need to load them)
            if ca.MX(v.nominal).is_zero() or ca.MX(v.nominal - 1).is_zero() or ca.MX(v.nominal + 1).is_zero():
                continue
            else:
                if ca.MX(v.nominal).size1() != 1:
                    logger.error('Vector Nominals not supported yet. ({})'.format(sym_name))
                self.__nominals[sym_name] = ca.fabs(v.nominal)
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("SimulationProblem: Setting nominal value for variable {} to {}".format(
                        sym_name, self.__nominals[sym_name]))

        # Initialize DAE and initial residuals
        variable_lists = ['states', 'der_states', 'alg_states', 'inputs', 'constants', 'parameters']
        function_arguments = [self.__pymoca_model.time] + [
            ca.veccat(*[v.symbol for v in getattr(self.__pymoca_model, variable_list)])
            for variable_list in variable_lists]

        self.__dae_residual = self.__pymoca_model.dae_residual_function(*function_arguments)

        self.__initial_residual = self.__pymoca_model.initial_residual_function(*function_arguments)
        if self.__initial_residual is None:
            self.__initial_residual = ca.MX()

        # Construct state vector
        self.__sym_list = self.__mx['states'] + self.__mx['algebraics'] + self.__mx['derivatives'] + \
            self.__mx['time'] + self.__mx['constant_inputs'] + self.__mx['parameters']
        self.__state_vector = np.full(len(self.__sym_list), np.nan)

        # A very handy index
        self.__states_end_index = len(self.__mx['states']) + \
            len(self.__mx['algebraics']) + len(self.__mx['derivatives'])

        # Construct a dict to look up symbols by name (or iterate over)
        self.__sym_dict = OrderedDict(((sym.name(), sym) for sym in self.__sym_list))

        # Assemble some symbolics, including those needed for a backwards Euler derivative approximation
        X = ca.vertcat(*self.__sym_list[:self.__states_end_index])
        X_prev = ca.vertcat(*[ca.MX.sym(sym.name() + '_prev') for sym in self.__sym_list[:self.__states_end_index]])
        dt = ca.MX.sym("delta_t")

        # Make a list of derivative approximations using backwards Euler formulation
        derivative_approximation_residuals = []
        for index, derivative_state in enumerate(self.__mx['derivatives']):
            derivative_approximation_residuals.append(derivative_state - (X[index] - X_prev[index]) / dt)

        # Append residuals for derivative approximations
        dae_residual = ca.vertcat(self.__dae_residual, *derivative_approximation_residuals)

        # TODO: implement lookup_tables

        # Make a list of unscaled symbols and a list of their scaled equivalent
        unscaled_symbols = []
        scaled_symbols = []
        for sym_name, nominal in self.__nominals.items():
            index = self.__get_state_vector_index(sym_name)
            # If the symbol is a state, Add the symbol to the lists
            if index <= self.__states_end_index:
                unscaled_symbols.append(X[index])
                scaled_symbols.append(X[index] * nominal)

                # Also scale previous states
                unscaled_symbols.append(X_prev[index])
                scaled_symbols.append(X_prev[index] * nominal)

        # Substitute unscaled terms for scaled terms
        dae_residual = ca.substitute(dae_residual, ca.vertcat(*unscaled_symbols), ca.vertcat(*scaled_symbols))

        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug('SimulationProblem: DAE Residual is ' + str(dae_residual))

        if X.size1() != dae_residual.size1():
            logger.error('Formulation Error: Number of states ({}) does not equal number of equations ({})'.format(
                X.size1(), dae_residual.size1()))

        # Construct function parameters
        parameters = ca.vertcat(dt, X_prev, *self.__sym_list[self.__states_end_index:])

        # Construct a function res_vals that returns the numerical residuals of a numerical state
        self.__res_vals = ca.Function("res_vals", [X, parameters], [dae_residual])

        # Use rootfinder() to make a function that takes a step forward in time by trying to zero res_vals()
        options = {'nlpsol': 'ipopt', 'nlpsol_options': self.solver_options()}
        self.__do_step = ca.rootfinder("next_state", "nlpsol", self.__res_vals, options)

        # Call parent class for default behaviour.
        super().__init__()

    def initialize(self, config_file=None):
        """
        Initialize state vector with default values

        :param config_file: Path to an initialization file.
        """
        if config_file:
            # TODO read start and stop time from config_file and call:
            # self.setup_experiment(start,stop)
            # for now, assume that setup_experiment was called beforehand
            raise NotImplementedError

        # Set values of parameters defined in the model into the state vector
        for var in self.__pymoca_model.parameters:
            # First check to see if parameter is already set (this allows child classes to override model defaults)
            if np.isfinite(self.get_var(var.symbol.name())):
                continue

            # Also test to see if the value is constant
            if isinstance(var.value, ca.MX) and not var.value.is_constant():
                continue

            # Try to extract the value
            try:
                # Extract the value as a python type
                val = var.python_type(var.value)
            except ValueError:
                # var.value is a float NaN being cast to non-float
                continue
            else:
                # If val is finite, we set it
                if np.isfinite(val):
                    logger.debug('SimulationProblem: Setting parameter {} = {}'.format(var.symbol.name(), val))
                    self.set_var(var.symbol.name(), val)

        # Assemble initial residuals and set values from start attributes into the state vector
        constrained_residuals = []
        minimized_residuals = []
        for var in itertools.chain(self.__pymoca_model.states, self.__pymoca_model.alg_states):
            var_name = var.symbol.name()
            var_nominal = self.get_variable_nominal(var_name)

            # Attempt to cast var.start to python type
            mx_start = ca.MX(var.start)
            if mx_start.is_constant():
                # cast var.start to python type
                start_val = var.python_type(mx_start.to_DM())
            else:
                # var.start is a symbolic expression with unknown value
                start_val = None

            if start_val == 0.0 and not var.fixed:
                # To make initialization easier, we allow setting initial states by providing timeseries
                # with names that match a symbol in the model. We only check for this matching if the start
                # and fixed attributes were left as default
                try:
                    start_val = self.initial_state()[var_name]
                except KeyError:
                    pass
                else:
                    # An initial state was found- add it to the constrained residuals
                    logger.debug('Initialize: Added {} = {} to initial equations (found matching timeseries).'.format(
                        var_name, start_val))
                    # Set var to be fixed
                    var.fixed = True

            # Attempt to set start_val in the state vector. Default to zero if unknown.
            try:
                self.set_var(var_name, start_val if start_val is not None else 0.0)
            except KeyError:
                logger.warning('Initialize: {} not found in state vector. Initial value of {} not set.'.format(
                    var_name, start_val))

            # Add a residual for the difference between the state and its starting expression
            start_expr = start_val if start_val is not None else var.start
            if var.fixed:
                # require residual = 0
                constrained_residuals.append((var.symbol - start_expr) / var_nominal)
            else:
                # minimize residual
                minimized_residuals.append((var.symbol - start_expr) / var_nominal)

        # Default start var for ders is zero
        for der_var in self.__mx['derivatives']:
            self.set_var(der_var.name(), 0.0)

        # Warn for nans in state vector (verify we didn't miss anything)
        self.__warn_for_nans()

        # Optionally encourage a steady-state initial condition
        if getattr(self, 'encourage_steady_state_initial_conditions', False):
            # add penalty for der(var) != 0.0
            for d in self.__mx['derivatives']:
                logger.debug('Added {} to the minimized residuals.'.format(d.name()))
                minimized_residuals.append(d)

        # Make minimized_residuals into a single symbolic object
        minimized_residual = ca.vertcat(*minimized_residuals)

        # Assemble symbolics needed to make a function describing the initial condition of the model
        # We constrain every entry in this MX to zero
        equality_constraints = ca.vertcat(self.__dae_residual, self.__initial_residual, *constrained_residuals)

        # The variables that need a mutually consistent initial condition
        X = ca.vertcat(*self.__sym_list[:self.__states_end_index])

        # Make a list of unscaled symbols and a list of their scaled equivalent
        unscaled_symbols = []
        scaled_symbols = []
        for sym_name, nominal in self.__nominals.items():
            # Add the symbol to the lists
            symbol = self.__sym_dict[sym_name]
            unscaled_symbols.append(symbol)
            scaled_symbols.append(symbol * nominal)

        # Make the lists symbolic
        unscaled_symbols = ca.vertcat(*unscaled_symbols)
        scaled_symbols = ca.vertcat(*scaled_symbols)

        # Substitute unscaled terms for scaled terms
        equality_constraints = ca.substitute(equality_constraints, unscaled_symbols, scaled_symbols)
        minimized_residual = ca.substitute(minimized_residual, unscaled_symbols, scaled_symbols)

        logger.debug('SimulationProblem: Initial Equations are ' + str(equality_constraints))
        logger.debug('SimulationProblem: Minimized Residuals are ' + str(minimized_residual))

        # State bounds can be symbolic, written in terms of parameters. After all
        # parameter values are known, we evaluate the numeric values of bounds.
        symbolic_bounds = ca.vertcat(*[ca.horzcat(v.min, v.max) for v in itertools.chain(
            self.__pymoca_model.states, self.__pymoca_model.alg_states, self.__pymoca_model.der_states)])
        bound_evaluator = ca.Function('bound_evaluator', self.__mx['parameters'], [symbolic_bounds])

        # Evaluate bounds using values of parameters
        n_parameters = len(self.__mx['parameters'])
        if n_parameters > 0:
            [evaluated_bounds] = bound_evaluator.call(self.__state_vector[-n_parameters:])
        else:
            [evaluated_bounds] = bound_evaluator.call([])

        # Construct arrays of state bounds (used in the initialize() nlp, but not in __do_step rootfinder)
        self.__lbx = evaluated_bounds[:, 0]
        self.__ubx = evaluated_bounds[:, 1]

        # Constrain model equation residuals to zero
        lbg = np.zeros(equality_constraints.size1())
        ubg = np.zeros(equality_constraints.size1())

        # Construct objective function from the input residual
        objective_function = ca.dot(minimized_residual, minimized_residual)

        # Construct nlp and solver to find initial state using ipopt
        parameters = ca.vertcat(*self.__mx['time'], *self.__mx['constant_inputs'], *self.__mx['parameters'])
        nlp = {'x': X, 'f': objective_function, 'g': equality_constraints, 'p': parameters}
        solver = ca.nlpsol('solver', 'ipopt', nlp, self.solver_options())

        # Construct guess
        guess = ca.vertcat(*np.nan_to_num(self.__state_vector[:self.__states_end_index]))

        # Find initial state
        initial_state = solver(x0=guess,
                               lbx=self.__lbx, ubx=self.__ubx,
                               lbg=lbg, ubg=ubg,
                               p=self.__state_vector[self.__states_end_index:])

        # If unsuccessful, stop.
        return_status = solver.stats()['return_status']
        if return_status not in {'Solve_Succeeded', 'Solved_To_Acceptable_Level'}:
            raise Exception('Initialization Failed with return status "{}"'.format(return_status))

        # Update state vector with initial conditions
        self.__state_vector[:self.__states_end_index] = initial_state['x'][:self.__states_end_index].T

        # make a copy of the initialized initial state vector in case we want to run the model again
        self.__initialized_state_vector = copy.deepcopy(self.__state_vector)

        # Warn for nans in state vector after initialization
        self.__warn_for_nans()

    def pre(self):
        """
        Any preprocessing takes place here.
        """
        pass

    def post(self):
        """
        Any postprocessing takes place here.
        """
        pass

    def setup_experiment(self, start, stop, dt):
        """
        Method for subclasses (PIMixin, CSVMixin, or user classes) to set timing information for a simulation run.

        :param start: Start time for the simulation.
        :param stop:  Final time for the simulation.
        :param dt:    Time step size.
        """

        # Set class vars with start/stop/dt values
        self.__start = start
        self.__stop = stop
        self.__dt = dt

        # Set time in state vector
        self.set_var('time', start)

    def update(self, dt):
        """
        Performs one timestep.

        The methods ``setup_experiment`` and ``initialize`` must have been called before.

        :param dt: Time step size.
        """
        if dt < 0:
            dt = self.__dt

        logger.debug("Taking a step at {} with size {}".format(self.get_current_time(), dt))

        # increment time
        self.set_var('time', self.get_current_time() + dt)

        # take a step
        guess = self.__state_vector[:self.__states_end_index]
        next_state = self.__do_step(guess, ca.vertcat(dt, *self.__state_vector))

        # make sure that the step converged sufficiently
        largest_res = ca.norm_inf(self.__res_vals(next_state, ca.vertcat(self.__dt, *self.__state_vector)))
        tol = self.solver_options().get('ipopt.tol', 1.0e-8)
        if largest_res > tol:
            logger.warning(
                'Simulation may have failed to converge at time {}. Residual value {} is greater than {}'.format(
                    self.get_current_time(), largest_res, tol))

        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug('Residual maximum magnitude: {:.2E}'.format(float(largest_res)))

        # Update state vector
        self.__state_vector[:self.__states_end_index] = next_state.T

    def simulate(self):
        """
        Run model from start_time to end_time.
        """

        # Do any preprocessing, which may include changing parameter values on
        # the model
        logger.info("Preprocessing")
        self.pre()

        # Initialize model
        logger.info("Initializing")
        self.initialize()

        # Perform all timesteps
        logger.info("Running")
        while self.get_current_time() < self.get_end_time():
            self.update(-1)

        # Do any postprocessing
        logger.info("Postprocessing")
        self.post()

    def reset(self):
        """
        Reset the FMU.
        """
        self.__state_vector = copy.deepcopy(self.__initialized_state_vector)

    def get_start_time(self):
        """
        Return start time of experiment.

        :returns: The start time of the experiment.
        """
        return self.__start

    def get_end_time(self):
        """
        Return end time of experiment.

        :returns: The end time of the experiment.
        """
        return self.__stop

    def get_current_time(self):
        """
        Return current time of simulation.

        :returns: The current simulation time.
        """
        return self.get_var('time')

    def get_time_step(self):
        """
        Return simulation timestep.

        :returns: The simulation timestep.
        """
        return self.__dt

    def get_var(self, name):
        """
        Return a numpy array from FMU.

        :param name: Variable name.

        :returns: The value of the variable.
        """

        # Get the canonical name and sign
        name, sign = self.alias_relation.canonical_signed(name)

        # Get the raw value of the canonical var
        index = self.__get_state_vector_index(name)
        value = self.__state_vector[index]

        # Adjust sign if needed
        if sign < 0:
            value *= sign

        # Adjust for nominal value if not default
        nominal = self.get_variable_nominal(name)
        if nominal != 1.0:
            value *= nominal

        return value

    def get_var_count(self):
        """
        Return the number of variables in the model.

        :returns: The number of variables in the model.
        """
        return len(self.get_variables())

    def get_var_name(self, i):
        """
        Returns the name of a variable.

        :param i: Index in ordered dictionary returned by method get_variables.

        :returns: The name of the variable.
        """
        return list(self.get_variables())[i]

    def get_var_type(self, name):
        """
        Return type, compatible with numpy.

        :param name: String variable name.

        :returns: The numpy-compatible type of the variable.

        :raises: KeyError
        """
        return self.__python_types[name]

    def get_var_rank(self, name):
        """
        Not implemented
        """
        raise NotImplementedError

    def get_var_shape(self, name):
        """
        Not implemented
        """
        raise NotImplementedError

    def get_variables(self):
        """
        Return all variables (both internal and user defined)

        :returns: An ordered dictionary of all variables supported by the model.
        """
        return self.__sym_dict

    @cached
    def get_state_variables(self):
        return AliasDict(
            self.alias_relation,
            {sym.name(): sym for sym in (self.__mx['states'] + self.__mx['algebraics'])})

    @cached
    def get_parameter_variables(self):
        return AliasDict(
            self.alias_relation,
            {sym.name(): sym for sym in self.__mx['parameters']})

    @cached
    def get_input_variables(self):
        return AliasDict(
            self.alias_relation,
            {sym.name(): sym for sym in self.__mx['constant_inputs']})

    @cached
    def get_output_variables(self):
        return self.__pymoca_model.outputs

    @cached
    def __get_state_vector_index(self, variable):
        index = next((i for i, sym in enumerate(self.__sym_list) if sym.name() == variable), None)
        if index is None:
            raise KeyError(str(variable) + " does not exist!")
        return index

    def __warn_for_nans(self):
        """
        Test state vector for missing values and warn
        """
        value_is_nan = np.isnan(self.__state_vector)
        if any(value_is_nan):
            for sym, isnan in zip(self.__sym_list, value_is_nan):
                if isnan:
                    logger.warning('Variable {} has no value.'.format(sym))

    def set_var(self, name, value):
        """
        Set the value of the given variable.

        :param name: Name of variable to set.
        :param value:  Value(s).
        """

        # TODO: sanitize input

        # Get the canonical name, adjust sign if needed
        name, sign = self.alias_relation.canonical_signed(name)
        if sign < 0:
            value *= sign

        # Adjust for nominal value if not default
        nominal = self.get_variable_nominal(name)
        if nominal != 1.0:
            value /= nominal

        # Store value in state vector
        index = self.__get_state_vector_index(name)
        self.__state_vector[index] = value

    def set_var_slice(self, name, start, count, var):
        """
        Not implemented.
        """
        raise NotImplementedError

    def set_var_index(self, name, index, var):
        """
        Not implemented.
        """
        raise NotImplementedError

    def inq_compound(self, name):
        """
        Not implemented.
        """
        raise NotImplementedError

    def inq_compound_field(self, name, index):
        """
        Not implemented.
        """
        raise NotImplementedError

    def solver_options(self):
        """
        Returns a dictionary of CasADi root_finder() solver options.

        :returns: A dictionary of CasADi :class:`root_finder` options.  See the CasADi documentation for details.
        """
        return {'ipopt.print_level': 0, 'print_time': False}

    def get_variable_nominal(self, variable):
        """
        Get the value of the nominal attribute of a variable
        """
        return self.__nominals.get(variable, 1.0)

    def timeseries_at(self, variable, t):
        """
        Get value of timeseries variable at time t: should be overridden by pi or csv mixin
        """
        raise NotImplementedError

    @cached
    def initial_state(self) -> AliasDict:
        """
        The initial state.

        :returns: A dictionary of variable names and initial state (t0) values.
        """
        t0 = self.get_start_time()
        initial_state_dict = AliasDict(self.alias_relation)

        for variable in list(self.get_state_variables()) + list(self.get_input_variables()):
            try:
                initial_state_dict[variable] = self.timeseries_at(variable, t0)
            except KeyError:
                pass
            except NotImplementedError:
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("Read intial state for {}".format(variable))

        return initial_state_dict

    @cached
    def parameters(self):
        """
        Return a dictionary of parameter values extracted from the Modelica model
        """
        # Create AliasDict
        parameters = AliasDict(self.alias_relation)

        # Update with model parameters
        parameters.update({p.symbol.name(): p.value for p in self.__pymoca_model.parameters})

        return parameters

    @property
    @cached
    def alias_relation(self):
        # Initialize aliases
        alias_relation = AliasRelation()
        for v in itertools.chain(self.__pymoca_model.states,
                                 self.__pymoca_model.der_states,
                                 self.__pymoca_model.alg_states,
                                 self.__pymoca_model.inputs):
            for alias in v.aliases:
                alias_relation.add(v.symbol.name(), alias)
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("SimulationProblem: Aliased {} to {}".format(
                        v.symbol.name(), alias))

        return alias_relation

    @cached
    def compiler_options(self):
        """
        Subclasses can configure the `pymoca <http://github.com/pymoca/pymoca>`_ compiler options here.

        :returns: A dictionary of pymoca compiler options.  See the pymoca documentation for details.
        """

        # Default options
        compiler_options = {}

        # Expand vector states to multiple scalar component states.
        compiler_options['expand_vectors'] = True

        # Where imported model libraries are located.
        library_folders = self.modelica_library_folders.copy()

        for ep in pkg_resources.iter_entry_points(group='rtctools.libraries.modelica'):
            if ep.name == "library_folder":
                library_folders.append(
                    pkg_resources.resource_filename(ep.module_name, ep.attrs[0]))

        compiler_options['library_folders'] = library_folders

        # Eliminate equations of the type 'var = const'.
        compiler_options['eliminate_constant_assignments'] = True

        # Eliminate constant symbols from model, replacing them with the values
        # specified in the model.
        compiler_options['replace_constant_values'] = True

        # Replace any constant expressions into the model.
        compiler_options['replace_constant_expressions'] = True

        # Replace any parameter expressions into the model.
        compiler_options['replace_parameter_expressions'] = True

        # Eliminate variables starting with underscores.
        compiler_options['eliminable_variable_expression'] = r'(.*[.]|^)_\w+\Z'

        # Automatically detect and eliminate alias variables.
        compiler_options['detect_aliases'] = True

        # Cache the model on disk
        compiler_options['cache'] = True

        # Done
        return compiler_options
