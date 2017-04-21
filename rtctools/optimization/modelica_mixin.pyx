# cython: embedsignature=True

from casadi import MX, substitute, repmat, vertcat
import numpy as np
import logging
import pyjmi
import sets
import os

from timeseries import Timeseries
from optimization_problem import OptimizationProblem, Alias
from casadi_helpers import resolve_interdependencies

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
        parameter_kinds = [self._jm_model.BOOLEAN_PARAMETER_DEPENDENT,
                           self._jm_model.BOOLEAN_PARAMETER_INDEPENDENT,
                           self._jm_model.INTEGER_PARAMETER_DEPENDENT,
                           self._jm_model.INTEGER_PARAMETER_INDEPENDENT,
                           self._jm_model.REAL_PARAMETER_INDEPENDENT,
                           self._jm_model.REAL_PARAMETER_DEPENDENT]

        self._mx['parameters'] = []
        for parameter_kind in parameter_kinds:
            self._mx['parameters'].extend([var.getVar() for var in self._jm_model.getVariables(
                parameter_kind) if not var.isAlias()])

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

        # Now condense equations
        self._condense_dae()

        # Call parent class first for default behaviour.
        super(ModelicaMixin, self).__init__(**kwargs)

    def _condense_dae(self):
        # This function condenses the DAE in such a way that unnecessary states are eliminated.

        logger.debug("ModelicaMixin: Condensing DAE")

        # An algebraic variable becomes a constraint residual candidate if it A) has numerical bounds and B) internal causality.
        # An algebraic variable is marked as private if it A) is unbounded, B) has internal causality, and C) starts with an underscore ('_').
        constraint_residual_candidates = []
        private_variables = []
        for var in self._jm_model.getVariables(self._jm_model.REAL_ALGEBRAIC):
            sym = var.getVar()
            if var.getCausality() == var.INTERNAL:
                m, M = None, None
                if var.hasAttributeSet('min'):
                    m = var.getAttribute('min')
                if var.hasAttributeSet('max'):
                    M = var.getAttribute('max')
                if m is not None or M is not None:
                    logger.debug("ModelicaMixin: Marking {} as a potential constraint residual.".format(sym.getName()))

                    constraint_residual_candidates.append((sym, m, M))
                else:
                    name = sym.getName()
                    if name.startswith('_') or name.startswith('temp_'):
                        logger.debug("ModelicaMixin: Marking {} as a private variable to be eliminated.".format(name))

                        private_variables.append(name)

        # Eliminate equations of the form x = y or z = f(x), where z is a private variable.
        dae = []
        dae_eq = []
        substitutions = {}
        algebraics_names = [sym.getName() for sym in self._mx['algebraics']]
        for eq in self._jm_model.getDaeEquations():
            lhs, rhs = eq.getLhs(), eq.getRhs()
            skip = False

            # This is an equation of the form x = y.  Create an alias, and substitute one with the other.
            if not skip:
                if lhs.isSymbolic() and rhs.isSymbolic():
                    if rhs.getName() in algebraics_names:
                        logger.debug("ModelicaMixin: Aliased {} to {}".format(rhs.getName(), lhs.getName()))
                        self._aliases[lhs.getName()] = self.variable_aliases(lhs.getName()) + self.variable_aliases(rhs.getName())
                        substitutions[rhs] = lhs
                        skip = True
                    elif lhs.getName() in algebraics_names:
                        logger.debug("ModelicaMixin: Aliased {} to {}".format(lhs.getName(), rhs.getName()))
                        self._aliases[rhs.getName()] = self.variable_aliases(rhs.getName()) + self.variable_aliases(lhs.getName())
                        substitutions[lhs] = rhs
                        skip = True

            # Look for equations of the form z = f(x), where z is a private variable.
            if not skip:
                if lhs.isSymbolic() and lhs.getName() in private_variables:
                    substitutions[lhs] = rhs
                    skip = True
                elif rhs.isSymbolic() and rhs.getName() in private_variables:
                    substitutions[rhs] = lhs
                    skip = True

            # Add equation, if it is not to be skipped.
            if skip:
                logger.debug("ModelicaMixin: Eliminating equation {} = {}".format(lhs, rhs))
            else:
                dae.append(lhs - rhs)
                dae_eq.append(eq)

        # Substitute eliminated variables z with f(x) in rest of DAE.
        logger.debug("ModelicaMixin: Substituting {} with {}".format(substitutions.keys(), substitutions.values()))

        self._mx['algebraics'] = list(sets.Set(self._mx['algebraics']) - sets.Set(substitutions.keys()))

        dae = substitute(dae, substitutions.keys(), substitutions.values())

        # Add path constraints for bounded, orphan algebraic residuals.
        self._path_constraints = []
        for constraint_residual_candidate in constraint_residual_candidates:
            matches = 0
            constraint_function = None
            constraint_eq = None
            for eq in dae_eq:
                lhs, rhs = eq.getLhs(), eq.getRhs()
                if lhs.isSymbolic():
                    if lhs.getName() == constraint_residual_candidate[0].getName():
                        constraint_function = rhs
                        constraint_eq = eq
                        matches += 1
                if rhs.isSymbolic():
                    if rhs.getName() == constraint_residual_candidate[0].getName():
                        constraint_function = lhs
                        constraint_eq = eq
                        matches += 1
                if matches > 1:
                    break
            if matches == 1:
                if constraint_function is not None:
                    # Remove from DAE
                    index = dae_eq.index(constraint_eq)
                    del dae[index]
                    del dae_eq[index]
                    self._mx['algebraics'].remove(constraint_residual_candidate[0])

                    # Add to constraints
                    m, M = constraint_residual_candidate[1], constraint_residual_candidate[2]
                    if m is None:
                        m = -np.inf
                    elif m.isConstant():
                        m = float(m)
                    else:
                        m_symbolic = True
                    if M is None:
                        M = np.inf
                    elif M.isConstant():
                        M = float(M)
                    else:
                        M_symbolic = True

                    if not m_symbolic and not M_symbolic:
                        constraint = (constraint_function, m, M)
                        logger.debug("ModelicaMixin: Adding constraint {} <= {} <= {}".format(constraint[1], constraint[0], constraint[2]))
                        self._path_constraints.append(constraint)
                    else:
                        if m_symbolic or np.isfinite(m):
                            constraint = (constraint_function - m, 0.0, np.inf)
                            logger.debug("ModelicaMixin: Adding constraint {} <= {} <= {}".format(constraint[1], constraint[0], constraint[2]))
                            self._path_constraints.append(constraint)

                        if M_symbolic or np.isfinite(M):
                            constraint = (constraint_function - M, -np.inf, 0.0)
                            logger.debug("ModelicaMixin: Adding constraint {} <= {} <= {}".format(constraint[1], constraint[0], constraint[2]))
                            self._path_constraints.append(constraint)

        # Store condensed DAE residual
        self._dae_residual = vertcat(dae)

        # Store condensed initial residual
        initial_residual = self._jm_model.getInitialResidual()
        [self._initial_residual] = substitute([initial_residual], substitutions.keys(), substitutions.values())

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
        return self._dae_residual

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
        for variable in self._mx['parameters']:
            variable = variable.getName()
            var = self._jm_model.getVariable(variable)
            if var.hasAttributeSet('bindingExpression'):
                parameters[variable] = var.getAttribute('bindingExpression')
                logger.debug("Read parameter {} from Modelica model".format(
                    variable))
            else:
                # Value will be provided by a subclass.
                pass

        return parameters

    def constant_inputs(self, ensemble_member):
        # Call parent class first for default values.
        constant_inputs = super(ModelicaMixin, self).constant_inputs(ensemble_member)

        # Return input values from JModelica model
        times = self.times()
        for variable in self._mx['constant_inputs']:
            variable = variable.getName()
            var = self._jm_model.getVariable(variable)
            if var.hasAttributeSet('bindingExpression'):
                value = var.getAttribute('bindingExpression')
                constant_inputs[variable] = Timeseries(
                    times, repmat([value], len(times)))
                logger.debug("Read constant input {} from Modelica model".format(
                    variable))
            else:
                # Value will be provided by a subclass.
                pass

        return constant_inputs

    def initial_state(self, ensemble_member):
        # Initial conditions obtained from "start=" get pulled into the initial
        # residual by JM.
        return {}

    @property
    def initial_residual(self):
        return self._initial_residual

    def bounds(self):
        # Call parent class first for default values.
        bounds = super(ModelicaMixin, self).bounds()

        # Load additional bounds from model
        # If a bound contains a parameter, we assume this parameter to be equal for all ensemble
        # members.
        parameters = self.parameters(0)        
        parameter_values = [None] * len(self.dae_variables['parameters'])
        values = []
        for i, symbol in enumerate(self.dae_variables['parameters']):
            found = False
            for alias in self.variable_aliases(symbol.getName()):
                if alias.name in parameters:
                    parameter_values[i] = alias.sign * parameters[alias.name]
                    found = True
                    break
            if not found:
                raise Exception("No value specified for parameter {}".format(symbol.getName()))
        parameter_values = resolve_interdependencies(parameter_values, self.dae_variables['parameters'])

        for variable in self._mx['states'] + self._mx['algebraics'] + self._mx['control_inputs']:
            variable = variable.getName()
            var = self._jm_model.getVariable(variable)
            if var.getType() == var.BOOLEAN:
                m, M = 0, 1
            else:
                m, M = None, None
            if var.hasAttributeSet('min'):
                [m] = substitute([var.getAttribute('min')], self.dae_variables['parameters'], parameter_values)
                m = float(m)
                if np.isnan(m):
                    m = -np.inf
            if var.hasAttributeSet('max'):
                [M] = substitute([var.getAttribute('max')], self.dae_variables['parameters'], parameter_values)
                M = float(M)
                if np.isnan(M):
                    M = np.inf
            bounds[variable] = (m, M)

        return bounds

    def variable_is_discrete(self, variable):
        var = self._jm_model.getVariable(variable)
        if var is None:
            return False
        else:
            return (var.getVariability() == var.DISCRETE)

    def path_constraints(self, ensemble_member):
        path_constraints = super(ModelicaMixin, self).path_constraints(ensemble_member)
        path_constraints.extend(self._path_constraints)
        return path_constraints

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
