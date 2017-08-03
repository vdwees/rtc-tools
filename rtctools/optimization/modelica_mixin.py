from pymola.backends.casadi.api import transfer_model
import casadi as ca
import numpy as np
import itertools
import logging
import os

from rtctools._internal.alias_tools import AliasRelation, AliasDict
from rtctools._internal.caching import cached
from rtctools._internal.casadi_helpers import substitute_in_external

from .timeseries import Timeseries
from .optimization_problem import OptimizationProblem

logger = logging.getLogger("rtctools")


class ModelicaMixin(OptimizationProblem):
    """
    Adds a `Modelica <http://www.modelica.org/>`_ model to your optimization problem.

    During preprocessing, the Modelica files located inside the ``model`` subfolder are loaded.

    :cvar modelica_library_folder: Folder in which any referenced Modelica libraries are to be found.  Default is ``mo``.
    """

    # Folder in which the referenced Modelica libraries are found
    modelica_library_folder = os.getenv('DELTARES_LIBRARY_PATH', 'mo')

    def __init__(self, **kwargs):
        # Check arguments
        assert('model_folder' in kwargs)

        # Transfer model from the Modelica .mo file to CasADi using pymola
        if 'model_name' in kwargs:
            model_name = kwargs['model_name']
        else:
            if hasattr(self, 'model_name'):
                model_name = self.model_name
            else:
                model_name = self.__class__.__name__

        self.__pymola_model = transfer_model(kwargs['model_folder'], model_name, self.compiler_options())

        # Extract the CasADi MX variables used in the model
        self.__mx = {}
        self.__mx['time'] = [self.__pymola_model.time]
        self.__mx['states'] = [v.symbol for v in self.__pymola_model.states]
        self.__mx['derivatives'] = [v.symbol for v in self.__pymola_model.der_states]
        self.__mx['algebraics'] = [v.symbol for v in self.__pymola_model.alg_states]
        self.__mx['parameters'] = [v.symbol for v in self.__pymola_model.parameters]
        self.__mx['control_inputs'] = []
        self.__mx['constant_inputs'] = []
        self.__mx['lookup_tables'] = []

        # Merge with user-specified delayed feedback
        delayed_feedback_variables = map(lambda delayed_feedback: delayed_feedback[
                                         1], self.delayed_feedback())

        for v in self.__pymola_model.inputs:
            if v.symbol.name() in delayed_feedback_variables:
                # Delayed feedback variables are local to each ensemble, and therefore belong to the collection of algebraic variables,
                # rather than to the control inputs.
                self.__mx['algebraics'].append(v.symbol)
            else:
                if v.symbol.name() in kwargs.get('lookup_tables', []):
                    self.__mx['lookup_tables'].append(v.symbol)
                elif v.fixed == True:
                    self.__mx['constant_inputs'].append(v.symbol)
                else:
                    self.__mx['control_inputs'].append(v.symbol)

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
                    logger.debug("ModelicaMixin: Set nominal value for variable {} to {}".format(
                        sym_name, self.__nominals[sym_name]))

            self.__python_types[sym_name] = v.python_type

        # Initialize dae and initial residuals
        # These are not in @cached dictionary properties so that we need to create the list
        # of function arguments only once.
        variable_lists = ['states', 'der_states', 'alg_states', 'inputs', 'constants', 'parameters']
        function_arguments = [self.__pymola_model.time] + [ca.veccat(*[v.symbol for v in getattr(self.__pymola_model, variable_list)]) for variable_list in variable_lists]

        self.__dae_residual = self.__pymola_model.dae_residual_function(*function_arguments)
        if self.__dae_residual is None:
            self.__dae_residual = ca.MX()

        self.__initial_residual = self.__pymola_model.initial_residual_function(*function_arguments)
        if self.__initial_residual is None:
            self.__initial_residual = ca.MX()

        # Log variables in debug mode
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug("ModelicaMixin: Found states {}".format(
                ', '.join([var.name() for var in self.__mx['states']])))
            logger.debug("ModelicaMixin: Found derivatives {}".format(
                ', '.join([var.name() for var in self.__mx['derivatives']])))
            logger.debug("ModelicaMixin: Found algebraics {}".format(
                ', '.join([var.name() for var in self.__mx['algebraics']])))
            logger.debug("ModelicaMixin: Found control inputs {}".format(
                ', '.join([var.name() for var in self.__mx['control_inputs']])))
            logger.debug("ModelicaMixin: Found constant inputs {}".format(
                ', '.join([var.name() for var in self.__mx['constant_inputs']])))
            logger.debug("ModelicaMixin: Found parameters {}".format(
                ', '.join([var.name() for var in self.__mx['parameters']])))

        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

    @cached
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
        compiler_options['cache'] = True

        # Done
        return compiler_options

    def delayed_feedback(self):
        delayed_feedback = super().delayed_feedback()
        delayed_feedback.extend([(dfb.origin, dfb.name, dfb.delay) for dfb in self.__pymola_model.delayed_states])
        return delayed_feedback

    @property
    def dae_residual(self):
        return self.__dae_residual

    @property
    def dae_variables(self):
        return self.__mx

    @property
    @cached
    def output_variables(self):
        output_variables = [v.symbol for v in self.__pymola_model.outputs]
        output_variables.extend(self.__mx['control_inputs'])
        return output_variables

    @cached
    def parameters(self, ensemble_member):
        # Call parent class first for default values.
        parameters = super().parameters(ensemble_member)

        # Return parameter values from pymola model
        for v in self.__pymola_model.parameters:
            parameters[v.symbol.name()] = v.value

        # Done
        return parameters

    @cached
    def constant_inputs(self, ensemble_member):
        # Call parent class first for default values.
        constant_inputs = super().constant_inputs(ensemble_member)

        # Return input values from pymola model
        times = self.times()
        constant_input_names = set(sym.name() for sym in self.__mx['constant_inputs'])
        for v in self.__pymola_model.inputs:
            if v.symbol.name() in constant_input_names:
                constant_inputs[v.symbol.name()] = Timeseries(
                    times, np.full_like(times, v.value))
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("Read constant input {} = {} from Modelica model".format(
                        v.symbol.name(), v.value))

        return constant_inputs

    @cached
    def history(self, ensemble_member):
        history = AliasDict(self.alias_relation)

        # Initial conditions obtained from start attributes.
        for v in self.__pymola_model.states:
            if v.fixed == True:
                history[v.symbol.name()] = Timeseries(np.array([self.initial_time]), np.array([v.start]))

        return history

    @cached
    def initial_state(self, ensemble_member):
        initial_state = AliasDict(self.alias_relation)

        # Initial conditions obtained from start attributes.
        for v in self.__pymola_model.states:
            if v.fixed == True:
                initial_state[v.symbol.name()] = v.start

        return initial_state

    @property
    def initial_residual(self):
        return self.__initial_residual

    @cached
    def bounds(self):
        # Call parent class first for default values.
        bounds = super().bounds()

        # Parameter values
        parameters = self.parameters(0)
        parameter_values = [parameters.get(param.name(), param) for param in self.__mx['parameters']]

        # Load additional bounds from model
        for v in itertools.chain(self.__pymola_model.states, self.__pymola_model.alg_states, self.__pymola_model.inputs):
            sym_name = v.symbol.name()
            
            try:
                (m, M) = bounds[sym_name]
            except KeyError:
                if self.__python_types.get(sym_name, float) == bool:
                    (m, M) = (0, 1)
                else:
                    (m, M) = (-np.inf, np.inf)

            m_ = ca.MX(v.min)
            if not m_.is_constant():
                [m_] = substitute_in_external([m_], self.__mx['parameters'], parameter_values)
                if not m_.is_constant():
                    raise Exception('Could not resolve lower bound for variable {}'.format(sym_name))
            m_ = float(m_)

            M_ = ca.MX(v.max)
            if not M_.is_constant():
                [M_] = substitute_in_external([M_], self.__mx['parameters'], parameter_values)
                if not M_.is_constant():
                    raise Exception('Could not resolve upper bound for variable {}'.format(sym_name))
            M_ = float(M_)

            # We take the intersection of all provided bounds
            m = max(m, m_)
            M = min(M, M_)

            bounds[sym_name] = (m, M)

        return bounds

    def variable_is__discrete(self, variable):
        return self.__python_types.get(variable, float) != float

    @property
    @cached
    def alias_relation(self):
        # Initialize aliases
        alias_relation = AliasRelation()
        for v in itertools.chain(self.__pymola_model.states, self.__pymola_model.der_states, self.__pymola_model.alg_states, self.__pymola_model.inputs):
            for alias in v.aliases:
                alias_relation.add(v.symbol.name(), alias)
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("ModelicaMixin: Aliased {} to {}".format(
                        v.symbol.name(), alias))

        return alias_relation

    def variable_nominal(self, variable):
        return self.__nominals.get(variable, 1)
