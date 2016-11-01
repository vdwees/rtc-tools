# cython: embedsignature=True

from casadi import MXFunction, MX, substitute
import numpy as np
import logging
import pyjmi
import os

from timeseries import Timeseries
from optimization_problem import OptimizationProblem, Alias

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

        # Transfer model from the Modelica .mo file to CasADi using JModelica
        if 'model_name' in kwargs:
            model_name = kwargs['model_name']
        else:
            if hasattr(self, 'model_name'):
                model_name = self.model_name
            else:
                model_name = self.__class__.__name__

        self._jm_model = pyjmi.transfer_model(model_name,
                                              [os.path.join(kwargs['model_folder'], f) for f in os.listdir(
                                                  kwargs['model_folder']) if f.endswith('.mo')],
                                              compiler_options=self.compiler_options())

        logger.debug("\n" + repr(self._jm_model))

        # Extract the CasADi MX variables used in the model
        state_vars = filter(lambda var: not var.isAlias(
        ), self._jm_model.getVariables(self._jm_model.DIFFERENTIATED))

        self._mx = {}
        self._mx['time'] = [self._jm_model.getTimeVariable()]
        self._mx['states'] = [var.getVar() for var in state_vars]
        self._mx['derivatives'] = [var.getMyDerivativeVariable().getVar()
                                   for var in state_vars]

        algebraic_kinds = [self._jm_model.REAL_ALGEBRAIC,
                           self._jm_model.REAL_DISCRETE,
                           self._jm_model.INTEGER_DISCRETE,
                           self._jm_model.BOOLEAN_DISCRETE]
        self._mx['algebraics'] = []
        for algebraic_kind in algebraic_kinds:
            self._mx['algebraics'].extend([var.getVar() for var in self._jm_model.getVariables(
                algebraic_kind) if not var.isAlias()])

        self._mx['control_inputs'] = []
        self._mx['constant_inputs'] = []
        self._mx['lookup_tables'] = []

        delayed_feedback_variables = map(lambda delayed_feedback: delayed_feedback[
                                         1], self.delayed_feedback())

        input_kinds = [self._jm_model.REAL_INPUT,
                       self._jm_model.INTEGER_INPUT,
                       self._jm_model.BOOLEAN_INPUT]
        for input_kind in input_kinds:
            for var in self._jm_model.getVariables(input_kind):
                sym = var.getVar()
                if sym.getName() in delayed_feedback_variables:
                    # Delayed feedback variables are local to each ensemble, and therefore belong to the collection of algebraic variables,
                    # rather than to the control inputs.
                    self._mx['algebraics'].append(sym)
                else:
                    if sym.getName() in kwargs.get('lookup_tables', []):
                        self._mx['lookup_tables'].append(sym)
                    elif bool(var.getAttribute('fixed')):
                        self._mx['constant_inputs'].append(sym)
                    else:
                        self._mx['control_inputs'].append(sym)

        self._output_variables = []
        for var in self._jm_model.getAllVariables():
            if var.getCausality() == var.OUTPUT:
                self._output_variables.append(var.getVar())
        self._output_variables.extend(self._mx['control_inputs'])

        # Initialize constants and parameters by eliminating them from the DAE
        # residual
        parameter_kinds = [self._jm_model.BOOLEAN_CONSTANT,
                           self._jm_model.BOOLEAN_PARAMETER_DEPENDENT,
                           self._jm_model.BOOLEAN_PARAMETER_INDEPENDENT,
                           self._jm_model.INTEGER_CONSTANT,
                           self._jm_model.INTEGER_PARAMETER_DEPENDENT,
                           self._jm_model.INTEGER_PARAMETER_INDEPENDENT,
                           self._jm_model.REAL_CONSTANT,
                           self._jm_model.REAL_PARAMETER_INDEPENDENT,
                           self._jm_model.REAL_PARAMETER_DEPENDENT]

        self._mx['parameters'] = []
        for parameter_kind in parameter_kinds:
            # Don't handle parameters starting with '_'.  These will generally
            # be settings that have been added by the compiler.
            self._mx['parameters'].extend([var.getVar() for var in self._jm_model.getVariables(
                parameter_kind) if not var.getName().startswith("_") and not var.isAlias()])

        # Output variables
        logger.debug("ModelicaMixin: Found states {}".format(
            ', '.join([var.getName() for var in self._mx['states']])))
        logger.debug("ModelicaMixin: Found derivatives {}".format(
            ', '.join([var.getName() for var in self._mx['derivatives']])))
        logger.debug("ModelicaMixin: Found algebraics {}".format(
            ', '.join([var.getName() for var in self._mx['algebraics']])))
        logger.debug("ModelicaMixin: Found control inputs {}".format(
            ', '.join([var.getName() for var in self._mx['control_inputs']])))
        logger.debug("ModelicaMixin: Found constant inputs {}".format(
            ', '.join([var.getName() for var in self._mx['constant_inputs']])))
        logger.debug("ModelicaMixin: Found parameters {}".format(
            ', '.join([var.getName() for var in self._mx['parameters']])))

        # Initialize aliases
        self._aliases = {}
        for var in self._jm_model.getAliases():
            model_var = var.getModelVariable()
            l = self._aliases.get(model_var.getName(), [
                                  Alias(model_var.getName(), False)])
            l.append(Alias(var.getName(), var.isNegated()))
            self._aliases[model_var.getName()] = l

            sign = ''
            if var.isNegated():
                sign = '-'
            logger.debug("ModelicaMixin: Aliased {} to {}{}".format(
                var.getName(), sign, model_var.getName()))

        # Initialize nominals
        self._nominals = {}
        for var in self._mx['states'] + self._mx['algebraics'] + self._mx['control_inputs']:
            nominal = self._jm_model.getVariable(var.getName()).getNominal()
            if nominal and nominal != 0:
                self._nominals[var.getName()] = abs(float(nominal))

                logger.debug("ModelicaMixin: Set nominal value for variable {} to {}".format(
                    var.getName(), self._nominals[var.getName()]))

        # Call parent class first for default behaviour.
        super(ModelicaMixin, self).__init__(**kwargs)

    def compiler_options(self):
        """
        Subclasses can configure the `JModelica.org <http://www.jmodelica.org/>`_ compiler options here.

        :returns: A dictionary of JModelica.org compiler options.  See the JModelica.org documentation for details.
        """

        # Default options
        compiler_options = {}

        # Don't automatically add initial equations.  The user generally provides initial conditions
        # through CSV or PI input files.
        compiler_options['automatic_add_initial_equations'] = False

        # We scale the model ourselves.
        compiler_options['enable_variable_scaling'] = False

        # No automatic division with variables please.  Our variables may
        # sometimes equal to zero.
        compiler_options['divide_by_vars_in_tearing'] = False

        # Don't propagate derivatives into equations by default, as it makes maching equations.
        # to states harder.
        compiler_options['propagate_derivatives'] = False

        # Include the 'mo' folder as library dir by default.
        compiler_options['extra_lib_dirs'] = self.modelica_library_folder

        # Disable index reduction and structural diagnosis by default, to allow
        # injection of splines into the model.
        compiler_options['index_reduction'] = False
        compiler_options['enable_structural_diagnosis'] = False

        # Done
        return compiler_options

    @property
    def dae_residual(self):
        # Extract the DAE residual
        return self._jm_model.getDaeResidual()

    @property
    def dae_variables(self):
        return self._mx

    @property
    def output_variables(self):
        return self._output_variables

    def parameters(self, ensemble_member):
        # Call parent class first for default values.
        parameters = super(ModelicaMixin, self).parameters(ensemble_member)

        # Return parameter values from JModelica model
        parameters = {}
        for parameter in self._mx['parameters']:
            try:
                parameters[parameter.getName()] = self._jm_model.get(
                    parameter.getName())
                logger.debug("Read parameter {} from Modelica model".format(
                    parameter.getName()))
            except RuntimeError:
                # We have a bindingExpression here (probably). Don't evaluate
                # it just yet; we may still be overriding its inputs in a
                # subclass.
                pass
        return parameters

    def constant_inputs(self, ensemble_member):
        # Call parent class first for default values.
        constant_inputs = super(ModelicaMixin, self).constant_inputs(ensemble_member)

        # Return input values from JModelica model
        constant_inputs = {}
        times = self.times()
        for variable in self._mx['constant_inputs']:
            try:
                constant_value = self._jm_model.get(variable.getName())
                constant_inputs[variable.getName()] = Timeseries(
                    times, constant_value * np.ones(len(times)))
                logger.debug("Read constant input {} from Modelica model".format(
                    variable.getName()))
            except RuntimeError:
                # We have a bindingExpression here (probably). Don't evaluate
                # it just yet; we may still be overriding its inputs in a
                # subclass.
                continue

        return constant_inputs

    def initial_state(self, ensemble_member):
        # Initial conditions obtained from "start=" get pulled into the initial
        # residual by JM.
        return {}

    @property
    def initial_residual(self):
        # Extract the initial residual
        return self._jm_model.getInitialResidual()

    def bounds(self):
        # Call parent class first for default values.
        bounds = super(ModelicaMixin, self).bounds()

        # Load additional bounds from model
        # If a bound contains a parameter, we assume this parameter to be equal for all ensemble
        # members.
        parameters = self.parameters(0)

        def substitute_parameters(attr):
            # Replace parameters and constant values
            # We only replace those for which we have values are available.
            symbols = []
            values = []
            for symbol in self.dae_variables['parameters']:
                for alias in self.variable_aliases(symbol.getName()):
                    if alias.name in parameters:
                        symbols.append(symbol)
                        values.append(alias.sign * parameters[alias.name])
                        break
            [val] = substitute([attr], symbols, values)
            if val.isConstant():
                return float(val)
            else:
                deps = [val.getDep(i).getName() for i in range(val.getNdeps())]
                raise Exception("Parameters with names {} not set.".format(deps))

        for sym in self._mx['states'] + self._mx['algebraics'] + self._mx['control_inputs']:
            variable = sym.getName()
            var = self._jm_model.getVariable(variable)
            if var.getType() == var.BOOLEAN:
                m, M = 0, 1
            else:
                m, M = None, None
            if var.hasAttributeSet('min'):
                m = substitute_parameters(var.getAttribute('min'))
            if var.hasAttributeSet('max'):
                M = substitute_parameters(var.getAttribute('max'))
            bounds[variable] = (m, M)
        return bounds

    def variable_is_discrete(self, variable):
        var = self._jm_model.getVariable(variable)
        if var == None:
            return False
        else:
            return (var.getVariability() == var.DISCRETE)

    def variable_aliases(self, variable):
        try:
            return self._aliases[variable]
        except KeyError:
            # We do not use setdefault() here, as we would then always allocate
            # the alias and the list, even if they would not be required.
            l = [Alias(variable, False)]
            self._aliases[variable] = l
            return l

    def variable_nominal(self, variable):
        return self._nominals.get(variable, 1)
