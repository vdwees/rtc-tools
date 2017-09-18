import os
import logging
import numpy as np
from datetime import timedelta
import bisect
import copy

import casadi as ca
import itertools
import pymola
import rtctools.data.csv as csv
import pymola.backends.casadi.api
from collections import OrderedDict


logger = logging.getLogger("rtctools")

import rtctools.data.rtc as rtc
import rtctools.data.pi as pi


class SimulationProblem:
    """
    Base class for all Simulation problems. Loads the Modelica Model.
    """

    # Folder in which the referenced Modelica libraries are found
    modelica_library_folder = os.getenv('DELTARES_LIBRARY_PATH', 'mo')

    def __init__(self, **kwargs):
        # Check arguments
        assert('model_folder' in kwargs)

        # Log pymola version
        logger.debug("Using pymola {}.".format(pymola.__version__))

        # Transfer model from the Modelica .mo file to CasADi using pymola
        if 'model_name' in kwargs:
            model_name = kwargs['model_name']
        else:
            if hasattr(self, 'model_name'):
                model_name = self.model_name
            else:
                model_name = self.__class__.__name__

        self.__pymola_model = pymola.backends.casadi.api.transfer_model(kwargs['model_folder'], model_name, self.compiler_options())

        # Extract the CasADi MX variables used in the model
        self.__mx = {}
        self.__mx['time'] = [self.__pymola_model.time]
        self.__mx['states'] = [v.symbol for v in self.__pymola_model.states]
        self.__mx['outputs'] = [v.symbol for v in self.__pymola_model.outputs]
        self.__mx['derivatives'] = [v.symbol for v in self.__pymola_model.der_states]
        self.__mx['algebraics'] = [v.symbol for v in self.__pymola_model.alg_states]
        self.__mx['parameters'] = [v.symbol for v in self.__pymola_model.parameters]
        self.__mx['constant_inputs'] = []
        self.__mx['lookup_tables'] = []

        # Merge with user-specified delayed feedback
        # TODO: get this working
        delayed_feedback_variables = [] #map(lambda delayed_feedback: delayed_feedback[1], self.delayed_feedback())

        for v in self.__pymola_model.inputs:
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


        # Initialize nominals and types
        # These are not in @cached dictionary properties for backwards compatibility.
        self.__nominals = {}
        self.__python_types = {}
        for v in itertools.chain(self.__pymola_model.states, self.__pymola_model.alg_states, self.__pymola_model.inputs):
            sym_name = v.symbol.name()
            # We need to take care to allow nominal vectors.
            if ca.MX(v.nominal).is_zero() or ca.MX(v.nominal - 1).is_zero():
                self.__nominals[sym_name] = ca.fabs(v.nominal)

                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("SimulationProblem: Set nominal value for variable {} to {}".format(
                        sym_name, self.__nominals[sym_name]))

            self.__python_types[sym_name] = v.python_type

        # Initialize dae and initial residuals
        variable_lists = ['states', 'der_states', 'alg_states', 'inputs', 'constants', 'parameters']
        function_arguments = [self.__pymola_model.time] + \
            [ca.veccat(*[v.symbol for v in getattr(self.__pymola_model, variable_list)]) for variable_list in variable_lists]

        self.__dae_residual = self.__pymola_model.dae_residual_function(*function_arguments)

        self.__initial_residual = self.__pymola_model.initial_residual_function(*function_arguments)
        if self.__initial_residual is None:
            self.__initial_residual = ca.MX()

        # Construct state vector
        self.__sym_iter = self.__mx['states'] + \
                          self.__mx['algebraics'] + \
                          self.__mx['time'] + \
                          self.__mx['constant_inputs'] + \
                          self.__mx['parameters']
        self.__state_vector = np.full(len(self.__sym_iter), np.nan)
        self.__states_end_index = len(self.__mx['states']) + len(self.__mx['algebraics'])
        self.__sym_dict = OrderedDict(((sym.name(), sym) for sym in self.__sym_iter))

        # Substitute ders for discrete states using backwards euler formulation
        X = ca.vertcat(*self.__sym_iter[:self.__states_end_index])
        X_prev = ca.MX.sym("prev_states", X.size1())
        dt = ca.MX.sym("delta_t")

        derivative_approximations = []
        for derivative_state in self.__mx['derivatives']:
            index = next((i for i, s in enumerate(X) if s.name() == derivative_state.name()[4:-1]))
            derivative_approximations.append((X[index] - X_prev[index]) / dt)

        derivatives = ca.vertcat(*self.__mx['derivatives'])
        derivative_approximations = ca.vertcat(*derivative_approximations)
        dae_residual_substituted_ders = ca.substitute(self.__dae_residual, derivatives, derivative_approximations)

        logger.debug('SimulationProblem: DAE Residual is ' + ', '.join(
            (str(res) for res in ca.vertsplit(dae_residual_substituted_ders))))

        if X.size1() != dae_residual_substituted_ders.size1():
            logger.error('Formulation Error: Number of states ({}) does not equal number of equations ({})'.format(
                X.size1(), dae_residual_substituted_ders.size1()))

        # TODO: implement lookup_tables
        # TODO: use alias relation (so the get and set api will work with aliases too)
        # TODO: output the variables with the output tag, and not one of their aliases

        # Construct function parameters
        parameters = ca.vertcat(dt, X_prev, *self.__sym_iter[self.__states_end_index:])

        # Construct a function res_vals that returns the numerical residuals of a numnerical state
        self.__res_vals = ca.Function("res_vals", [X, parameters], [dae_residual_substituted_ders])

        # Use rootfinder() to make a function that takes a step forward in time by trying to zero res_vals()
        self.__do_step = ca.rootfinder("next_state", "newton", self.__res_vals, self.solver_options())

        # Call parent class first for default behaviour.
        super().__init__()

    def initialize(self):
        """
        Initialize state vector with default values

        """

        initial_residual = self.__initial_residual
        dae_residual = self.__dae_residual
        symbol_dict = self.__sym_dict

        # Assemble residual for start attributes 
        start_attribute_residuals = []
        for state in self.__pymola_model.states:
            if not state.fixed:
                if type(state.start) == ca.MX:
                    # state.start is a symbol from the model, so we attempt to
                    # set it equal to the value of that symbol 
                    # TODO: does this make sense:
                    # self.set(state.symbol.name(), self.get(state.start.name()))?
                    pass
                else:
                    # state.start has a numerical value, so we set it in the state vector
                    self.set_var(state.symbol.name(), state.start)
            else:
                # add a residual for the difference between the state and its starting value
                start_attribute_residuals.append(symbol_dict[state.symbol.name()]-state.start)

        # make a function describing the initial contition
        full_initial_residual = ca.vertcat(dae_residual, initial_residual, *start_attribute_residuals)
        X = ca.vertcat(*self.__sym_iter[:self.__states_end_index], *self.__mx['derivatives'])
        parameters = ca.vertcat(*self.__sym_iter[self.__states_end_index:])

        logger.debug('SimulationProblem: Initial Residual is ' +', '.join(
            (str(res) for res in ca.vertsplit(full_initial_residual))))

        if X.size1() != full_initial_residual.size1():
            logger.error('Initialization Error: Number of states ({}) does not equal number of initial equations ({})'.format(
                X.size1(), full_initial_residual.size1()))

        # Use rootfinder() to construct a function to find consistant intial conditions
        f = ca.Function("initial_residual", [X, parameters], [full_initial_residual])
        find_initial_state = ca.rootfinder("find_initial_state", "newton", f, self.solver_options())

        # Convert any np.nan (unset values) to a default guess of 0.0 and get the initial state
        guess = ca.vertcat(*np.nan_to_num(self.__state_vector[:self.__states_end_index]), *np.zeros_like(self.__mx['derivatives']))
        initial_state = find_initial_state(guess, ca.vertcat(self.__state_vector[self.__states_end_index:]))

        # Update state vector with initial conditions
        self.__state_vector[:self.__states_end_index] = initial_state[:self.__states_end_index].T

        # make a copy of the initialized initial state vector in case we want to run the model again
        self.__initial_state_vector = copy.deepcopy(self.__state_vector)

        # Test state vector for missing values and warn
        value_is_nan = np.isnan(self.__state_vector)
        if any(value_is_nan):
            for sym, isnan in zip(self.__sym_iter, value_is_nan):
                if isnan:
                    logger.warning('Variable {} has no value.'.format(sym))

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

    def get_current_residual_values(self):
        """
        Returns the residual values (equation error) of the current state
        """
        return self.__res_vals(self.__state_vector[:self.__states_end_index], ca.vertcat(self.__dt, *self.__state_vector))

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
        self.__state_vector = copy.deepcopy(self.__initial_state_vector)

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

    def get_options(self):
        """
        Return the available options of the FMU.

        :returns: A dictionary of options supported by the FMU.
        """
        raise NotImplementedError
        return self.__model.simulate_options()

    def get_var(self, name):
        """
        Return a numpy array from FMU.

        :param name: Variable name.

        :returns: The value of the variable.
        """

        index = self.__get_state_vector_index(name)
        return self.__state_vector[index]

    def get_var_count(self):
        """
        Return the number of variables (internal FMU and user declared).

        :returns: The number of variables supported by the FMU.
        """
        return len(self.get_model_variables())

    def get_var_name(self, i):
        """
        Returns the name of a variable.

        :param i: Index in ordered dictionary returned by FMU-method get_model_variables.

        :returns: The name of the variable.
        """
        return self.get_variables().items()[i][0]

    def get_var_type(self, name):
        """
        Return type string, compatible with numpy.

        :param name: Variable name.

        :returns: The type of the variable.
        """
        raise NotImplementedError
        retval = self.__model.get_variable_data_type(name)
        return self.__model_types[retval]

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
        Return all variables of FMU (both internal and user defined)

        :returns: A list of all variables supported by the FMU.
        """
        return self.__sym_dict

    def get_parameter_variables(self):
        return {sym.name(): sym for sym in self.__mx['parameters']}

    def get_input_variables(self):
        return {sym.name(): sym for sym in self.__mx['constant_inputs']}

    def get_output_variables(self):
        return {sym.name(): sym for sym in self.__mx['outputs']}

    def __get_state_vector_index(self, variable):
        # TODO: cache these indices
        index =  next((i for i, sym in enumerate(self.__sym_iter) if sym.name() == variable), None)
        if index is None:
            raise KeyError(str(variable) + " does not exist!")
        return index

    def set_var(self, name, value):
        """
        Set the value of the given variable.

        :param name: Name of variable to set.
        :param value:  Value(s).
        """

        # TODO: sanitize input
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
        return dict(abstol = 1e-10,
                    linear_solver = 'csparse')

    def compiler_options(self):
        """
        Subclasses can configure the `pymola <http://github.com/jgoppert/pymola>`_ compiler options here.

        :returns: A dictionary of pymola compiler options.  See the pymola documentation for details.
        """

        # Default options
        compiler_options = {}

        # Expand vector states to multiple scalar component states.
        compiler_options['expand_vectors'] = True

        # Where imported model libraries are located.
        compiler_options['library_folders'] = [self.modelica_library_folder]

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
        compiler_options['eliminable_variable_expression'] = r'_\w+'

        # Automatically detect and eliminate alias variables.
        compiler_options['detect_aliases'] = True

        # Cache the model on disk
        compiler_options['cache'] = False #TODO: fix file suffix error when caching

        # Done
        return compiler_options