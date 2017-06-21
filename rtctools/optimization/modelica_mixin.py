# cython: embedsignature=True

from casadi import MX, substitute, repmat, vertcat, depends_on, veccat
from pymola.backends.casadi.api import transfer_model
from collections import OrderedDict
import numpy as np
import itertools
import logging
import os

from .timeseries import Timeseries
from .optimization_problem import OptimizationProblem
from .alias_tools import AliasRelation, AliasDict
from .casadi_helpers import substitute_in_external
from .caching import cached

logger = logging.getLogger("rtctools")


class ModelicaMixin(OptimizationProblem):
    """
    Adds a `Modelica <http://www.modelica.org/>`_ model to your optimization problem.

    During preprocessing, the Modelica files located inside the ``model`` subfolder are loaded.

    :cvar modelica_library_folder: Folder in which any referenced Modelica libraries are to be found.  Default is ``mo``.
    """

    # Folder in which the referenced Modelica libraries are found
    modelica_library_folder = os.getenv('DELTARES_LIBRARY_PATH', 'mo')

    def _symbols(self, l):
        return [v.symbol for v in l]

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

        self._pymola_model = transfer_model(kwargs['model_folder'], model_name, self.compiler_options())

        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug("\n" + repr(self._pymola_model))

        # Extract the CasADi MX variables used in the model
        self._mx = {}
        self._mx['time'] = [self._pymola_model.time]
        self._mx['states'] = [v.symbol for v in self._pymola_model.states]
        self._mx['derivatives'] = [v.symbol for v in self._pymola_model.der_states]
        self._mx['algebraics'] = [v.symbol for v in self._pymola_model.alg_states]
        self._mx['parameters'] = [v.symbol for v in self._pymola_model.parameters]
        self._mx['control_inputs'] = []
        self._mx['constant_inputs'] = []
        self._mx['lookup_tables'] = []

        # Merge with user-specified delayed feedback
        delayed_feedback_variables = map(lambda delayed_feedback: delayed_feedback[
                                         1], self.delayed_feedback())

        for v in self._pymola_model.inputs:
            if v.symbol.name() in delayed_feedback_variables:
                # Delayed feedback variables are local to each ensemble, and therefore belong to the collection of algebraic variables,
                # rather than to the control inputs.
                self._mx['algebraics'].append(v.symbol)
            else:
                if v.symbol.name() in kwargs.get('lookup_tables', []):
                    self._mx['lookup_tables'].append(v.symbol)
                elif v.fixed == True:
                    self._mx['constant_inputs'].append(v.symbol)
                else:
                    self._mx['control_inputs'].append(v.symbol)

        self._output_variables = [v.symbol for v in self._pymola_model.outputs]
        self._output_variables.extend(self._mx['control_inputs'])

        # Output variables
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug("ModelicaMixin: Found states {}".format(
                ', '.join([var.name() for var in self._mx['states']])))
            logger.debug("ModelicaMixin: Found derivatives {}".format(
                ', '.join([var.name() for var in self._mx['derivatives']])))
            logger.debug("ModelicaMixin: Found algebraics {}".format(
                ', '.join([var.name() for var in self._mx['algebraics']])))
            logger.debug("ModelicaMixin: Found control inputs {}".format(
                ', '.join([var.name() for var in self._mx['control_inputs']])))
            logger.debug("ModelicaMixin: Found constant inputs {}".format(
                ', '.join([var.name() for var in self._mx['constant_inputs']])))
            logger.debug("ModelicaMixin: Found parameters {}".format(
                ', '.join([var.name() for var in self._mx['parameters']])))

        # Initialize aliases
        self._aliases = {}
        self._alias_relation = AliasRelation()
        for v in itertools.chain(self._pymola_model.states, self._pymola_model.der_states, self._pymola_model.alg_states, self._pymola_model.inputs):
            for alias in v.aliases:
                self._alias_relation.add(v.symbol.name(), alias)
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("ModelicaMixin: Aliased {} to {}".format(
                        v.symbol.name(), alias))

        # Initialize nominals and types
        self._nominals = {}
        self._discrete = {}
        for v in itertools.chain(self._pymola_model.states, self._pymola_model.alg_states, self._pymola_model.inputs):
            sym_name = v.symbol.name()  
            if v.nominal != 0 and v.nominal != 1:
                self._nominals[sym_name] = abs(float(v.nominal))

                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("ModelicaMixin: Set nominal value for variable {} to {}".format(
                        sym_name, self._nominals[sym_name]))

            self._discrete[sym_name] = v.python_type != float

        # Initialize dae and initial residuals
        inputs = [v.symbol for v in self._pymola_model.inputs]

        self._dae_residual = self._pymola_model.dae_residual_function(self._mx['time'][0],
            veccat(*self._mx['states']), veccat(*self._mx['derivatives']), veccat(*self._mx['algebraics']), veccat(*inputs), MX(), veccat(*self._mx['parameters']))
        if self._dae_residual is None:
            self._dae_residual = MX()

        self._initial_residual = self._pymola_model.initial_residual_function(self._mx['time'][0],
            veccat(*self._mx['states']), veccat(*self._mx['derivatives']), veccat(*self._mx['algebraics']), veccat(*inputs), MX(), veccat(*self._mx['parameters']))
        if self._initial_residual is None:
            self._initial_residual = MX()

        # Call parent class first for default behaviour.
        super(ModelicaMixin, self).__init__(**kwargs)

    @cached
    def compiler_options(self):
        """
        Subclasses can configure the `pymola <http://github.com/jgoppert/pymola>`_ compiler options here.

        :returns: A dictionary of pymola compiler options.  See the pymola documentation for details.
        """

        # Default options
        compiler_options = {}

        # Where imported model libraries are located.
        compiler_options['library_folders'] = [self.modelica_library_folder]

        # Eliminate constant symbols from model, replacing them with the values
        # specified in the model.
        compiler_options['replace_constant_values'] = True

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
        delayed_feedback = super(ModelicaMixin, self).delayed_feedback()
        delayed_feedback.extend([(dfb.origin, dfb.name, dfb.delay) for dfb in self._pymola_model.delayed_states])
        return delayed_feedback

    @property
    def dae_residual(self):
        return self._dae_residual

    @property
    def dae_variables(self):
        return self._mx

    @property
    def output_variables(self):
        return self._output_variables

    @cached
    def parameters(self, ensemble_member):
        # Call parent class first for default values.
        parameters = super(ModelicaMixin, self).parameters(ensemble_member)

        # Return parameter values from pymola model
        for v in self._pymola_model.parameters:
            parameters[v.symbol.name()] = v.value

        # Done
        return parameters

    @cached
    def constant_inputs(self, ensemble_member):
        # Call parent class first for default values.
        constant_inputs = super(ModelicaMixin, self).constant_inputs(ensemble_member)

        # Return input values from pymola model
        times = self.times()
        constant_input_names = set(sym.name() for sym in self._mx['constant_inputs'])
        for v in self._pymola_model.inputs:
            if v.symbol.name() in constant_input_names:
                constant_inputs[v.symbol.name()] = Timeseries(
                    times, np.full_like(times, v.value))
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("Read constant input {} = {} from Modelica model".format(
                        sym.name(), sym.value))

        return constant_inputs

    @cached
    def history(self, ensemble_member):
        history = AliasDict(self._alias_relation)

        # Initial conditions obtained from start attributes.
        for v in self._pymola_model.states:
            if v.fixed == True:
                history[v.symbol.name()] = Timeseries(np.array([self.initial_time]), np.array([v.start]))

        return history

    @cached
    def initial_state(self, ensemble_member):
        initial_state = AliasDict(self._alias_relation)

        # Initial conditions obtained from start attributes.
        for v in self._pymola_model.states:
            if v.fixed == True:
                initial_state[v.symbol.name()] = v.start

        return initial_state

    @property
    def initial_residual(self):
        return self._initial_residual

    @cached
    def bounds(self):
        # Call parent class first for default values.
        bounds = super(ModelicaMixin, self).bounds()

        # Parameter values
        parameters = self.parameters(0)
        parameter_values = [parameters.get(param.name(), param) for param in self._mx['parameters']]

        # Load additional bounds from model
        for v in itertools.chain(self._pymola_model.states, self._pymola_model.alg_states, self._pymola_model.inputs):
            sym_name = v.symbol.name()
            (m, M) = bounds.get(sym_name, (None, None))
            if self._discrete.get(sym_name, False):
                if m is None:
                    m = 0
                if M is None:
                    M = 1

            m_ = MX(v.min)
            if not m_.is_constant():
                [m] = substitute_in_external([m_], self._mx['parameters'], parameter_values)
                if m.is_constant():
                    m = float(m)
            else:
                m_ = float(m_)
                if np.isfinite(m_): # TODO vector values
                    m = m_

            M_ = MX(v.max)
            if not M_.is_constant():
                [M] = substitute_in_external([M_], self._mx['parameters'], parameter_values)
                if M.is_constant():
                    M = float(M)
            else:
                M_ = float(M_)
                if np.isfinite(M_):
                    M = M_

            bounds[sym_name] = (m, M)

        return bounds

    def variable_is_discrete(self, variable):
        return self._discrete.get(variable, False)

    @property
    def alias_relation(self):
        return self._alias_relation

    def variable_nominal(self, variable):
        return self._nominals.get(variable, 1)
