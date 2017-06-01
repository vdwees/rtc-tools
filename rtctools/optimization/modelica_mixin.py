# cython: embedsignature=True

from pymola import gen_casadi
from casadi import MX, substitute, repmat, vertcat, depends_on
from collections import OrderedDict
import numpy as np
import itertools
import logging
import sets
import os

from .timeseries import Timeseries
from .optimization_problem import OptimizationProblem, Alias
from .alias_tools import AliasRelation, AliasDict
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

        self._pymola_model = pymola.gen_casadi.load_model(model_name,
                                              [os.path.join(kwargs['model_folder'], f) for f in os.listdir(
                                                  kwargs['model_folder']) if f.endswith('.mo')],
                                              compiler_options=self.compiler_options())

        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug("\n" + repr(self._pymola_model))

        # Extract the CasADi MX variables used in the model
        self._mx = {}
        self._mx['time'] = [self._pymola_model.time]
        self._mx['states'] = self._pymola_model.states
        self._mx['derivatives'] = self._pymola_model.der_states
        self._mx['algebraics'] = self._pymola_model.alg_states
        self._mx['parameters'] = self._pymola_model.parameters
        self._mx['control_inputs'] = []
        self._mx['constant_inputs'] = []
        self._mx['lookup_tables'] = []

        # TODO inputs and algebraics overlap

        # TODO obsolete
        delayed_feedback_variables = map(lambda delayed_feedback: delayed_feedback[
                                         1], self.delayed_feedback())

        for sym in self._pymola_model.inputs:
            if sym.name() in delayed_feedback_variables:
                # Delayed feedback variables are local to each ensemble, and therefore belong to the collection of algebraic variables,
                # rather than to the control inputs.
                self._mx['algebraics'].append(sym)
            else:
                if sym.name() in kwargs.get('lookup_tables', []):
                    self._mx['lookup_tables'].append(sym)
                elif getattr(sym, 'fixed', False) == False:
                    self._mx['constant_inputs'].append(sym)
                else:
                    self._mx['control_inputs'].append(sym)

        self._output_variables = self._pymola_model.outputs
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
        for sym in itertools.join(*self._mx.values()):
            # TODO use alias relation
            for alias in getattr(sym, 'aliases', []):
                self._alias_relation.add(sym.name(), alias)
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("ModelicaMixin: Aliased {} to {}".format(
                        sym.name(), alias))

        # Initialize nominals and types
        self._nominals = {}
        self._discrete = {}
        for sym in itertools.join(*self._mx.values()):  
            sym_name = sym.name()  
            nominal = getattr(sym, 'nominal', None)
            if nominal and nominal != 0:
                self._nominals[sym_name] = abs(float(nominal))

                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("ModelicaMixin: Set nominal value for variable {} to {}".format(
                        sym_name, self._nominals[sym_name]))

            self._discrete[sym_name] = getattr(sym, 'type', float) != float

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

        # Eliminate constant symbols from model, replacing them with the values
        # specified in the model.
        compiler_options['replace_constants'] = True

        # Replace any parameter expressions into the model.
        compiler_options['replace_parameter_expressions'] = 'True'

        # Eliminate variables starting with underscores.
        compiler_options['eliminable_variable_expression'] = r'_\w+'

        # Automatically detect and eliminate alias variables.
        compiler_options['detect_aliases'] = True

        # Cache the model on disk
        compiler_options['cache'] = True

        # Done
        return compiler_options

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
        for sym in self._mx['parameters']:
            if hasattr(sym, 'value'):
                parameters[sym.name()] = sym.value

        # Done
        return parameters

    @cached
    def constant_inputs(self, ensemble_member):
        # Call parent class first for default values.
        constant_inputs = super(ModelicaMixin, self).constant_inputs(ensemble_member)

        # Return input values from pymola model
        times = self.times()
        for sym in self._mx['constant_inputs']:
            if hasattr(sym, 'value'):
                constant_inputs[sym.name()] = Timeseries(
                    times, repmat(sym.value, len(times)))
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("Read constant input {} = {} from Modelica model".format(
                        sym.name(), sym.value))

        return constant_inputs

    @cached
    def initial_state(self, ensemble_member):
        initial_state = AliasDict(self._alias_relation)

        # Initial conditions obtained from start attributes.
        for sym in self._mx['states']:
            if hasattr(sym, 'start'):
                initial_state[sym.name()] = sym.start

        return initial_state

    @property
    def initial_residual(self):
        return self._initial_residual

    @cached
    def bounds(self):
        # Call parent class first for default values.
        bounds = super(ModelicaMixin, self).bounds()

        # Parameter values TODO
        parameters = self.parameters(0)
        parameter_values = [parameters.get(param.name(), param) for param in self._mx['parameters']]

        # Load additional bounds from model
        for sym in itertools.chain(self._mx['states'], self._mx['algebraics'], self._mx['control_inputs'], self._eliminated_algebraics):
            sym_name = sym.name()
            (m, M) = bounds.get(sym_name, (None, None))
            if getattr(sym, 'type', float) == bool:
                if m is None:
                    m = 0
                if M is None:
                    M = 1
            if hasattr(sym, 'min'):
                m_ = sym.min
                if not m_.isConstant():
                    [m] = substitute([m_], self._mx['parameters'], parameter_values)
                    if m.isConstant():
                        m = float(m)
                else:
                    m_ = float(m_)
                    if np.isfinite(m_):
                        m = m_
            if hasattr(sym, 'max'):
                M_ = sym.max
                if not M_.isConstant():
                    [M] = substitute([M_], self._mx['parameters'], parameter_values)
                    if M.isConstant():
                        M = float(M)
                else:
                    M_ = float(M_)
                    if np.isfinite(M_):
                        M = M_
            bounds[variable] = (m, M)

        return bounds

    def variable_is_discrete(self, variable):
        return self._discrete.get(variable, False)

    @property
    def alias_relation(self):
        return self._alias_relation

    def variable_nominal(self, variable):
        return self._nominals.get(variable, 1)
