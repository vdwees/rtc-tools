# cython: embedsignature=True

from casadi import MX, MXFunction, ImplicitFunction, nlpIn, nlpOut, vertcat, horzcat, vec, substitute, sumRows, sumCols, interp1d, transpose, repmat, dependsOn, reshape, mul
from abc import ABCMeta, abstractmethod
import numpy as np
import itertools
import logging

from optimization_problem import OptimizationProblem
from timeseries import Timeseries
from casadi_helpers import *

logger = logging.getLogger("rtctools")


class CollocatedIntegratedOptimizationProblem(OptimizationProblem):
    """
    Discretizes your model using a mixed collocation/integration scheme.

    Collocation means that the discretized model equations are included as constraints
    between state variables in the optimization problem.

    .. note::

        To ensure that your optimization problem only has globally optimal solutions,
        any model equations that are collocated must be linear.  By default, all
        model equations are collocated, and linearity of the model equations is
        verified.  Working with non-linear models is possible, but discouraged.

    :cvar check_collocation_linearity: If ``True``, check whether collocation constraints are linear.  Default is ``True``.
    """

    __metaclass__ = ABCMeta

    #: Check whether the collocation constraints are linear
    check_collocation_linearity = True

    def __init__(self, **kwargs):
        # Variables that will be optimized
        self.dae_variables['free_variables'] = self.dae_variables[
            'states'] + self.dae_variables['algebraics'] + self.dae_variables['control_inputs']

        # Cache names of states
        self._differentiated_states = [
            variable.getName() for variable in self.dae_variables['states']]
        self._algebraic_states = [variable.getName()
                                  for variable in self.dae_variables['algebraics']]
        self._controls = [variable.getName()
                          for variable in self.dae_variables['control_inputs']]

    @abstractmethod
    def times(self, variable=None):
        pass

    @property
    def integrated_states(self):
        """
        A list of states that are integrated rather than collocated.

        .. warning:: This is an experimental feature.
        """
        return []

    @property
    def theta(self):
        """
        RTC-Tools discretizes differential equations of the form

        .. math::

            \dot{x} = f(x, u)

        using the :math:`\\theta`-method

        .. math::

            x_{i+1} = x_i + \Delta t \\left[\\theta f(x_{i+1}, u_{i+1}) + (1 - \\theta) f(x_i, u_i)\\right]

        The default is :math:`\\theta = 1`, resulting in the implicit or backward Euler method.  Note that in this
        case, the control input at the initial time step is not used.

        Set :math:`\\theta = 0` to use the explicit or forward Euler method.  Note that in this
        case, the control input at the final time step is not used.

        .. warning:: This is an experimental feature for :math:`0 < \\theta < 1`.
        """

        # Default to implicit Euler collocation, which is cheaper to evaluate than the trapezoidal method, while being A-stable.
        # N.B.  Setting theta to 0 will cause problems with algebraic equations, unless a consistent initialization is supplied for the algebraics.
        # N.B.  Setting theta to any value strictly between 0 and 1 will cause algebraic equations to be solved in an average sense.  This may
        #       induce unexpected oscillations.
        # TODO Fix these issue by performing index reduction and splitting DAE into ODE and algebraic parts.
        #      Theta then only applies to the ODE part.
        return 1.0

    def transcribe(self):
        # DAE residual
        dae_residual = self.dae_residual

        # Initial residual
        initial_residual = self.initial_residual

        logger.info("Transcribing problem with a DAE of {} equations, {} collocation points, and {} free variables".format(
            dae_residual.size1(), len(self.times()), len(self.dae_variables['free_variables'])))

        # Cache path variable names
        self._path_variable_names = [variable.getName()
                                     for variable in self.path_variables]

        # Initialize control discretization
        control_size, discrete_control, lbx_control, ubx_control, x0_control = self.discretize_controls()

        # Initialize state discretization
        state_size, discrete_state, lbx_state, ubx_state, x0_state = self.discretize_states()

        # Initialize vector of optimization symbols
        X = MX.sym('X', control_size + state_size)
        self._solver_input = X

        # Initialize bound and seed vectors
        discrete = np.zeros(X.size1(), dtype=np.bool)

        lbx = -np.inf * np.ones(X.size1())
        ubx = np.inf * np.ones(X.size1())

        x0 = np.zeros(X.size1())

        discrete[:len(discrete_control)] = discrete_control
        discrete[len(discrete_control):] = discrete_state
        lbx[:len(lbx_control)] = lbx_control
        lbx[len(lbx_control):] = lbx_state
        ubx[:len(ubx_control)] = ubx_control
        ubx[len(lbx_control):] = ubx_state
        x0[:len(x0_control)] = x0_control
        x0[len(x0_control):] = x0_state

        # Path objective
        path_objective = self.path_objective(0)

        # Path constraints
        path_constraints = self.path_constraints(0)
        path_constraint_expressions = vertcat([f_constraint for (f_constraint, lb, ub) in path_constraints])

        # Delayed feedback
        delayed_feedback = self.delayed_feedback()

        # Initial time
        t0 = self.initial_time

        # Establish integrator theta
        theta = self.theta

        # Provide a state for self.state_at() to work with.
        self._control_size = control_size
        self._state_size = state_size
        self._symbol_cache = {}

        # Insert lookup tables.  No support yet for different lookup tables per ensemble member.
        lookup_tables = self.lookup_tables(0)
        inserted_lookup_tables = set()
        
        for sym in self.dae_variables['lookup_tables']:
            found = False
            sym_key = None
            for alias in self.variable_aliases(sym.getName()):
                if alias.name in lookup_tables:
                    sym_key = alias.name
                    inserted_lookup_tables.add(alias.name)
                    found = True
                    break
            if not found:
                raise Exception(
                    "Unable to find lookup table function for {}".format(sym.getName()))

            input_syms = []
            for input in lookup_tables[sym_key].inputs:
                found = False
                input_sym = None
                for symbol in self.dae_variables['free_variables']:
                    for alias in self.variable_aliases(symbol.getName()):
                        if alias.name == input.getName():
                            input_sym = symbol
                            found = True
                            break
                if not found:
                    raise Exception(
                        "Unable to find input symbol {} in model".format(input.getName()))
                input_syms.append(input_sym)

            [value] = lookup_tables[sym_key].function(input_syms)
            [dae_residual] = substitute(
                [dae_residual], [sym], [value])

        if len(self.dae_variables['lookup_tables']) > 0 and self.ensemble_size > 1:
            logger.warning("Using lookup tables of ensemble member #0 for all members.")

        # Free variables for the collocated optimization problem
        integrated_variables = []
        collocated_variables = []
        for variable in itertools.chain(self.dae_variables['states'], self.dae_variables['algebraics']):
            if variable.getName() in self.integrated_states:
                integrated_variables.append(variable)
            else:
                collocated_variables.append(variable)
        for variable in self.dae_variables['control_inputs']:
            # TODO treat these separately.
            collocated_variables.append(variable)

        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug("Integrating variables {}".format(
                repr(integrated_variables)))
            logger.debug("Collocating variables {}".format(
                repr(collocated_variables)))

        # Split derivatives into "integrated" and "collocated" lists.
        integrated_derivatives = []
        collocated_derivatives = []
        for k, var in enumerate(self.dae_variables['states']):
            if var.getName() in self.integrated_states:
                integrated_derivatives.append(
                    self.dae_variables['derivatives'][k])
            else:
                collocated_derivatives.append(
                    self.dae_variables['derivatives'][k])
        self._algebraic_and_control_derivatives = []
        for k, var in enumerate(itertools.chain(self.dae_variables['algebraics'], self.dae_variables['control_inputs'])):
            sym = MX.sym('der({})'.format(var.getName()))
            self._algebraic_and_control_derivatives.append(sym)
            collocated_derivatives.append(sym)

        # Collocation times
        collocation_times = self.times()
        n_collocation_times = len(collocation_times)

        # Create a store of all ensemble-member-specific data for all ensemble members
        ensemble_store = [{} for i in range(self.ensemble_size)] # N.B. Don't use n * [{}], as it creates n refs to the same dict.
        for ensemble_member in range(self.ensemble_size):
            ensemble_data = ensemble_store[ensemble_member]

            # Store parameters
            parameters = self.parameters(ensemble_member)
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
            ensemble_data["parameters"] = nullvertcat(parameter_values)

            # Store constant inputs
            constant_inputs = self.constant_inputs(ensemble_member)
            constant_inputs_interpolated = {}
            for variable in self.dae_variables['constant_inputs']:
                found = False
                for alias in self.variable_aliases(variable.getName()):
                    if alias.name in constant_inputs:
                        constant_input = constant_inputs[alias.name]
                        values = constant_input.values
                        if isinstance(values, MX):
                            [values] = substitute([values], self.dae_variables['parameters'], parameter_values)
                        elif np.any([not MX(value).isConstant() for value in values]):
                            values = substitute(values, self.dae_variables['parameters'], parameter_values)
                        constant_inputs_interpolated[variable.getName()] = alias.sign * self.interpolate(
                            collocation_times, constant_input.times, values, 0.0, 0.0)
                        found = True
                        break
                if not found:
                    raise Exception("No values found for constant input {}".format(variable.getName()))
            ensemble_data["constant_inputs"] = constant_inputs_interpolated

            # Store initial state and derivatives
            initial_state = []
            initial_derivatives = []
            for variable in integrated_variables + collocated_variables:
                variable = variable.getName()
                value = self.state_vector(
                    variable, ensemble_member=ensemble_member)[0]
                nominal = self.variable_nominal(variable)
                if nominal != 1:
                    value *= nominal
                initial_state.append(value)
                initial_derivatives.append(self.der_at(
                    variable, t0, ensemble_member=ensemble_member))
            ensemble_data["initial_state"] = vertcat(initial_state)
            ensemble_data["initial_derivatives"] = vertcat(initial_derivatives)

            # Store initial path variables
            initial_path_variables = []
            for j, variable in enumerate(self.path_variables):
                variable = variable.getName()
                values = self.state_vector(
                    variable, ensemble_member=ensemble_member)
                initial_path_variables.append(values[0])
            ensemble_data["initial_path_variables"] = nullvertcat(initial_path_variables)

        # Replace parameters which are constant across the entire ensemble
        ensemble_const_parameters = []
        ensemble_const_parameter_values = []

        ensemble_parameters = []
        ensemble_parameter_values = [[] for i in range(self.ensemble_size)]

        for i, parameter in enumerate(self.dae_variables['parameters']):
            values = [ensemble_store[ensemble_member]["parameters"][i] for ensemble_member in range(self.ensemble_size)]
            if np.min(values) == np.max(values):
                ensemble_const_parameters.append(parameter)
                ensemble_const_parameter_values.append(values[0])
            else:
                ensemble_parameters.append(parameter)
                for ensemble_member in range(self.ensemble_size):
                    ensemble_parameter_values[ensemble_member].append(values[ensemble_member])

        [dae_residual, initial_residual, path_objective, path_constraint_expressions] = \
            substitute([dae_residual, initial_residual, path_objective, path_constraint_expressions],
                ensemble_const_parameters, ensemble_const_parameter_values)

        ensemble_parameters = vertcat(ensemble_parameters)

        # Aggregate ensemble data
        ensemble_aggregate = {}
        ensemble_aggregate["parameters"] = horzcat([nullvertcat(l) for l in ensemble_parameter_values])
        ensemble_aggregate["initial_constant_inputs"] = horzcat([nullvertcat([float(d["constant_inputs"][variable.getName()][0]) 
            for variable in self.dae_variables['constant_inputs']]) for d in ensemble_store])
        ensemble_aggregate["initial_state"] = horzcat([d["initial_state"] for d in ensemble_store])
        ensemble_aggregate["initial_state"] = reduce_matvec(ensemble_aggregate["initial_state"], self.solver_input)
        ensemble_aggregate["initial_derivatives"] = horzcat([d["initial_derivatives"] for d in ensemble_store])
        ensemble_aggregate["initial_derivatives"] = reduce_matvec(ensemble_aggregate["initial_derivatives"], self.solver_input)
        ensemble_aggregate["initial_path_variables"] = horzcat([d["initial_path_variables"] for d in ensemble_store])
        ensemble_aggregate["initial_path_variables"] = reduce_matvec(ensemble_aggregate["initial_path_variables"], self.solver_input)

        # Split DAE into integrated and into a collocated part
        dae_residual_integrated = []
        dae_residual_collocated = []
        for output_index in range(dae_residual.size1()):
            output = dae_residual[output_index]

            contains = False
            for derivative in integrated_derivatives:
                if dependsOn(output, derivative):
                    contains = True
                    break

            if contains:
                dae_residual_integrated.append(output)
            else:
                dae_residual_collocated.append(output)
        dae_residual_integrated = vertcat(dae_residual_integrated)
        dae_residual_collocated = vertcat(dae_residual_collocated)

        # Check linearity of collocated part
        self._affine_collocation_constraints = True
        if self.check_collocation_linearity and dae_residual_collocated.size1() > 0:
            # Check linearity of collocation constraints, which is a necessary condition for the optimization problem to be convex
            classification = classify_linear(dae_residual_collocated, vertcat(
                collocated_variables + integrated_variables + collocated_derivatives + integrated_derivatives))
            for j in range(len(classification)):
                # TODO detect conditionals!
                if classification[j] == 2:
                    self._affine_collocation_constraints = False

                    logger.warning(
                        "The DAE equation {} is non-linear.  The optimization problem is not convex.  This will, in general, result in the existence of multiple local optima and trouble finding a feasible initial solution.".format(dae_residual_collocated[j]))

        # Transcribe DAE using theta method collocation
        f = []
        g = []
        lbg = []
        ubg = []

        if len(integrated_variables) > 0:
            I = MX.sym('I', len(integrated_variables))
            I0 = MX.sym('I0', len(integrated_variables))
            C0 = [MX.sym('C0[{}]'.format(i)) for i in range(len(collocated_variables))]
            CI0 = [MX.sym('CI0[{}]'.format(i)) for i in range(len(self.dae_variables['constant_inputs']))]
            dt_sym = MX.sym('dt')

            integrated_finite_differences = (I - I0) / dt_sym

            [dae_residual_integrated_0] = substitute([dae_residual_integrated], integrated_variables + collocated_variables + integrated_derivatives + self.dae_variables['constant_inputs'] + self.dae_variables['time'], [I0[i] for i in range(len(integrated_variables))] + [
                                                     C0[i] for i in range(len(collocated_variables))] + [integrated_finite_differences[i] for i in range(len(integrated_derivatives))] + [CI0[i] for i in range(len(self.dae_variables['constant_inputs']))] + [self.dae_variables['time'][0] - dt_sym])
            [dae_residual_integrated_1] = substitute([dae_residual_integrated], integrated_variables + integrated_derivatives, [
                                                     I[i] for i in range(len(integrated_variables))] + [integrated_finite_differences[i] for i in range(len(integrated_derivatives))])

            if theta == 0:
                dae_residual_integrated = dae_residual_integrated_0
            elif theta == 1:
                dae_residual_integrated = dae_residual_integrated_1
            else:
                dae_residual_integrated = (
                    1 - theta) * dae_residual_integrated_0 + theta * dae_residual_integrated_1

            dae_residual_function_integrated = MXFunction('dae_residual_function_integrated', [I, I0, ensemble_parameters, vertcat([C0[i] for i in range(len(collocated_variables))] + [CI0[i] for i in range(len(
                self.dae_variables['constant_inputs']))] + [dt_sym] + collocated_variables + collocated_derivatives + self.dae_variables['constant_inputs'] + self.dae_variables['time'])], [dae_residual_integrated])
            # Expand to SX for improved performance
            # We do not expand the overall problem, as that would unroll
            # map/mapaccum as well into an SX tree.
            dae_residual_function_integrated = dae_residual_function_integrated.expand()

            options = self.integrator_options()
            integrator_step_function = ImplicitFunction('integrator_step_function', 'newton', dae_residual_function_integrated, options)

        # Initialize an MXFunction for the DAE residual (collocated part)
        if len(collocated_variables) > 0:
            dae_residual_function_collocated = MXFunction('dae_residual_function_collocated', [ensemble_parameters, vertcat(
                integrated_variables + collocated_variables + integrated_derivatives + collocated_derivatives + self.dae_variables['constant_inputs'] + self.dae_variables['time'])], [dae_residual_collocated])
            # Expand to SX for improved performance
            # We do not expand the overall problem, as that would unroll
            # map/mapaccum as well into an SX tree.
            dae_residual_function_collocated = dae_residual_function_collocated.expand()

        # Initialize an MXFunction for the path objective
        # Note that we assume that the path objective expression is the same for all ensemble members
        path_objective_function = MXFunction('path_objective',
                                               [ensemble_parameters,
                                                vertcat(integrated_variables + collocated_variables + integrated_derivatives + collocated_derivatives + self.dae_variables[
                                                        'constant_inputs'] + self.dae_variables['time'] + self.path_variables),
                                                X],
                                               [path_objective])
        path_objective_function = path_objective_function.expand()

        # Initialize an MXFunction for the path constraints
        # Note that we assume that the path constraint expression is the same for all ensemble members
        path_constraints_function = MXFunction('path_constraints',
                                               [ensemble_parameters,
                                                vertcat(integrated_variables + collocated_variables + integrated_derivatives + collocated_derivatives + self.dae_variables[
                                                        'constant_inputs'] + self.dae_variables['time'] + self.path_variables),
                                                X],
                                               [path_constraint_expressions])
        path_constraints_function = path_constraints_function.expand()

        # Set up accumulation over time (integration, and generation of
        # collocation constraints)
        if len(integrated_variables) > 0:
            # When using mapaccum, we also feed back the current
            # collocation constraints through accumulated_X.
            accumulated_X = MX.sym('accumulated_X', len(
                integrated_variables) + dae_residual_collocated.size1() + len(path_constraints) + 1)
        else:
            accumulated_X = MX.sym('accumulated_X', 0)
        accumulated_U = MX.sym('accumulated_U', 2 * (len(collocated_variables) + len(
            self.dae_variables['constant_inputs']) + 1) + len(self.path_variables))

        integrated_states_0 = accumulated_X[0:len(integrated_variables)]
        integrated_states_1 = MX.sym(
            'integrated_states_1', len(integrated_variables))
        collocated_states_0 = accumulated_U[0:len(collocated_variables)]
        collocated_states_1 = accumulated_U[
            len(collocated_variables):2 * len(collocated_variables)]
        constant_inputs_0 = accumulated_U[2 * len(collocated_variables):2 * len(
            collocated_variables) + len(self.dae_variables['constant_inputs'])]
        constant_inputs_1 = accumulated_U[2 * len(collocated_variables) + len(self.dae_variables[
            'constant_inputs']):2 * len(collocated_variables) + 2 * len(self.dae_variables['constant_inputs'])]
        collocation_time_0 = accumulated_U[
            2 * (len(collocated_variables) + len(self.dae_variables['constant_inputs'])) + 0]
        collocation_time_1 = accumulated_U[
            2 * (len(collocated_variables) + len(self.dae_variables['constant_inputs'])) + 1]
        path_variables_1 = accumulated_U[
            2 * (len(collocated_variables) + len(self.dae_variables['constant_inputs']) + 1):]

        # Approximate derivatives using backwards finite differences
        dt = collocation_time_1 - collocation_time_0
        collocated_finite_differences = (
            collocated_states_1 - collocated_states_0) / dt

        # We use vertcat to compose the list into an MX.  This is, in
        # CasADi 2.4, faster.
        accumulated_Y = []

        # Integrate integrated states
        if len(integrated_variables) > 0:
            # Perform step by computing implicit function
            # CasADi shares subexpressions that are bundled into the same Function.
            # The first argument is the guess for the new value of
            # integrated_states.
            [integrated_states_1] = integrator_step_function([integrated_states_0,
                                                              integrated_states_0,
                                                              ensemble_parameters,
                                                              vertcat([collocated_states_0,
                                                                       constant_inputs_0,
                                                                       dt,
                                                                       collocated_states_1,
                                                                       collocated_finite_differences,
                                                                       constant_inputs_1,
                                                                       collocation_time_1 - t0])],
                                                             False, True)
            accumulated_Y.append(integrated_states_1)

            # Recompute finite differences with computed new state, for use in the collocation part below
            # We don't use substititute() for this, as it becomes expensive
            # over long integration horizons.
            if len(collocated_variables) > 0:
                integrated_finite_differences = (
                    integrated_states_1 - integrated_states_0) / dt
        else:
            integrated_finite_differences = MX()

        # Call DAE residual at collocation point
        # Time stamp following paragraph 3.6.7 of the Modelica
        # specifications, version 3.3.
        if len(collocated_variables) > 0:
            if theta < 1:
                # Obtain state vector
                [dae_residual_0] = dae_residual_function_collocated([ensemble_parameters,
                                                                     vertcat([integrated_states_0,
                                                                              collocated_states_0,
                                                                              integrated_finite_differences,
                                                                              collocated_finite_differences,
                                                                              constant_inputs_0,
                                                                              collocation_time_0 - t0])],
                                                                    False, True)
            if theta > 0:
                # Obtain state vector
                [dae_residual_1] = dae_residual_function_collocated([ensemble_parameters,
                                                                     vertcat([integrated_states_1,
                                                                              collocated_states_1,
                                                                              integrated_finite_differences,
                                                                              collocated_finite_differences,
                                                                              constant_inputs_1,
                                                                              collocation_time_1 - t0])],
                                                                    False, True)
            if theta == 0:
                accumulated_Y.append(dae_residual_0)
            elif theta == 1:
                accumulated_Y.append(dae_residual_1)
            else:
                accumulated_Y.append(
                    (1 - theta) * dae_residual_0 + theta * dae_residual_1)

        accumulated_Y.extend(path_objective_function([ensemble_parameters,
                                                      vertcat([integrated_states_1,
                                                               collocated_states_1,
                                                               integrated_finite_differences,
                                                               collocated_finite_differences,
                                                               constant_inputs_1,
                                                               collocation_time_1 - t0,
                                                               path_variables_1]),
                                                      X],
                                                       False, True))

        accumulated_Y.extend(path_constraints_function([ensemble_parameters,
                                                        vertcat([integrated_states_1,
                                                                 collocated_states_1,
                                                                 integrated_finite_differences,
                                                                 collocated_finite_differences,
                                                                 constant_inputs_1,
                                                                 collocation_time_1 - t0,
                                                                 path_variables_1]),
                                                        X],
                                                       False, True))

        # Use map/mapaccum to capture integration and collocation constraint generation over the entire
        # time horizon with one symbolic operation.  This saves a lot of
        # memory.
        accumulated = MXFunction('accumulated',
            [accumulated_X, accumulated_U, ensemble_parameters, X], [vertcat(accumulated_Y)])

        if len(integrated_variables) > 0:
            accumulation = accumulated.mapaccum(
                'accumulation', n_collocation_times - 1)
        else:
            # Fully collocated problem.  Use map(), so that we can use
            # parallelization along the time axis.
            accumulation = accumulated.map(
                'accumulation', n_collocation_times - 1, {'parallelization': 'openmp'})

        # Add constraints for initial conditions
        initial_residual_with_params_fun = MXFunction('initial_residual', [ensemble_parameters, vertcat(self.dae_variables['states'] + self.dae_variables['algebraics'] + self.dae_variables[
                                                  'control_inputs'] + integrated_derivatives + collocated_derivatives + self.dae_variables['constant_inputs'] + self.dae_variables['time'])], [vertcat([dae_residual, initial_residual])])
        # Expand to SX for improved performance
        initial_residual_with_params_fun = initial_residual_with_params_fun.expand()

        initial_residual_with_params_fun_map = initial_residual_with_params_fun.map('initial_residual_with_params_fun_map', self.ensemble_size)
        [res] = initial_residual_with_params_fun_map([ensemble_aggregate["parameters"], vertcat([ensemble_aggregate["initial_state"], ensemble_aggregate["initial_derivatives"], ensemble_aggregate["initial_constant_inputs"], repmat([0.0], 1, self.ensemble_size)])], False, True)

        res = vec(res)
        g.append(res)
        zeros = [0.0] * res.size1()
        lbg.extend(zeros)
        ubg.extend(zeros)

        # Process the objectives and constraints for each ensemble member separately.
        # Note that we don't use map here for the moment, so as to allow each ensemble member to define its own
        # constraints and objectives.  Path constraints are applied for all ensemble members simultaneously
        # at the moment.  We can get rid of map again, and allow every ensemble member to specify its own
        # path constraints as well, once CasADi has some kind of loop detection.
        for ensemble_member in range(self.ensemble_size):
            logger.info("Transcribing ensemble member {}/{}".format(ensemble_member + 1, self.ensemble_size))

            initial_state = ensemble_aggregate["initial_state"][:, ensemble_member]
            initial_derivatives = ensemble_aggregate["initial_derivatives"][:, ensemble_member]
            initial_path_variables = ensemble_aggregate["initial_path_variables"][:, ensemble_member]
            initial_constant_inputs = ensemble_aggregate["initial_constant_inputs"][:, ensemble_member]
            parameters = ensemble_aggregate["parameters"][:, ensemble_member]

            constant_inputs = ensemble_store[ensemble_member]["constant_inputs"]

            # Initial conditions specified in history timeseries
            history = self.history(ensemble_member)
            for state in history.keys():
                try:
                    history_timeseries = history[state]
                    xinit = self.interpolate(
                        t0, history_timeseries.times, history_timeseries.values, np.nan, np.nan)

                except KeyError:
                    xinit = np.nan

                if np.isfinite(xinit):
                    # Avoid the use of slow state_at().  We don't need
                    # interpolation or history values here.
                    value = None
                    for variable in self.dae_variables['free_variables']:
                        variable = variable.getName()
                        for alias in self.variable_aliases(variable):
                            if alias.name == state:
                                value = self.state_vector(
                                    variable, ensemble_member=ensemble_member)[0]
                                nominal = self.variable_nominal(variable)
                                if nominal != 1:
                                    value *= nominal
                                if alias.sign < 0:
                                    value *= -1
                                break
                    if value is None:
                        # This was no free variable.
                        continue
                    g.append(value)
                    lbg.append(float(xinit))
                    ubg.append(float(xinit))
                else:
                    logger.warning("No ")

            # Initial conditions for integrator
            accumulation_X0 = []
            for variable in self.integrated_states:
                value = self.state_vector(
                    variable, ensemble_member=ensemble_member)[0]
                nominal = self.variable_nominal(variable)
                if nominal != 1:
                    value *= nominal
                accumulation_X0.append(value)
            if len(self.integrated_states) > 0:
                accumulation_X0.extend([0.0] * (dae_residual_collocated.size1() + 1))
            accumulation_X0 = vertcat(accumulation_X0)

            # Input for map
            logger.info("Interpolating states")

            accumulation_U = [None] * (1 + 2 * len(self.dae_variables['constant_inputs']) + 3)

            interpolated_states = [None] * (2 * len(collocated_variables))
            for j, variable in enumerate(collocated_variables):
                variable = variable.getName()
                times = self.times(variable)
                values = self.state_vector(
                    variable, ensemble_member=ensemble_member)
                if len(collocation_times) != len(times):
                    interpolated = interp1d(
                        times, values, collocation_times, self.equidistant)
                else:
                    interpolated = values
                nominal = self.variable_nominal(variable)
                if nominal != 1:
                    interpolated *= nominal
                interpolated_states[j] = interpolated[0:n_collocation_times - 1]
                interpolated_states[len(collocated_variables) +
                               j] = interpolated[1:n_collocation_times]
            accumulation_U[0] = reduce_matvec(horzcat(interpolated_states), self.solver_input)

            for j, variable in enumerate(self.dae_variables['constant_inputs']):
                variable = variable.getName()
                constant_input = constant_inputs[variable]
                accumulation_U[
                    1 + j] = MX(constant_input[0:n_collocation_times - 1])
                accumulation_U[1 + len(self.dae_variables[
                    'constant_inputs']) + j] = MX(constant_input[1:n_collocation_times])

            accumulation_U[1 + 2 * len(self.dae_variables[
                'constant_inputs'])] = MX(collocation_times[0:n_collocation_times - 1])
            accumulation_U[1 + 2 * len(self.dae_variables[
                'constant_inputs']) + 1] = MX(collocation_times[1:n_collocation_times])

            path_variables = [None] * len(self.path_variables)
            for j, variable in enumerate(self.path_variables):
                variable = variable.getName()
                values = self.state_vector(
                    variable, ensemble_member=ensemble_member)
                path_variables[j] = values[1:n_collocation_times]
            accumulation_U[1 + 2 * len(
                self.dae_variables['constant_inputs']) + 2] = reduce_matvec(horzcat(path_variables), self.solver_input)

            # Construct matrix using O(states) CasADi operations
            # This is faster than using blockcat, presumably because of the
            # row-wise scaling operations.
            logger.info("Aggregating and de-scaling variables")     

            accumulation_U = transpose(horzcat(accumulation_U))

            # Map to all time steps
            logger.info("Mapping")

            [integrators_and_collocation_and_path_constraints] = accumulation(
                [accumulation_X0, accumulation_U, repmat(parameters, 1, n_collocation_times - 1), repmat(X, 1, n_collocation_times - 1)])
            if integrators_and_collocation_and_path_constraints.size2() > 0:
                integrators = integrators_and_collocation_and_path_constraints[:len(integrated_variables), :]
                collocation_constraints = vec(integrators_and_collocation_and_path_constraints[len(integrated_variables):len(
                    integrated_variables) + dae_residual_collocated.size1(), 0:n_collocation_times - 1])
                discretized_path_objective = vec(integrators_and_collocation_and_path_constraints[len(
                    integrated_variables) + dae_residual_collocated.size1():len(
                    integrated_variables) + dae_residual_collocated.size1() + path_objective.size1(), 0:n_collocation_times - 1])
                discretized_path_constraints = vec(integrators_and_collocation_and_path_constraints[len(
                    integrated_variables) + dae_residual_collocated.size1() + path_objective.size1():, 0:n_collocation_times - 1])
            else:
                integrators = MX()
                collocation_constraints = MX()
                discretized_path_objective = MX()
                discretized_path_constraints = MX()

            logger.info("Composing NLP segment")

            # Store integrators for result extraction
            if len(integrated_variables) > 0:
                self.integrators = {}
                for i, variable in enumerate(integrated_variables):
                    self.integrators[variable.getName()] = integrators[i, :]
                self.integrators_mx = []
                for j in range(integrators.size2()):
                    self.integrators_mx.append(integrators[:, j])

            # Add collocation constraints
            if collocation_constraints.size1() > 0:
                #if self._affine_collocation_constraints:
                #    collocation_constraints = reduce_matvec_plus_b(collocation_constraints, self.solver_input)
                g.append(collocation_constraints)
                zeros = collocation_constraints.size1() * [0.0]
                lbg.extend(zeros)
                ubg.extend(zeros)

            # Delayed feedback
            # TODO implement for integrated states too, but first wait for
            # delay() support in JModelica.
            for (out_variable_name, in_variable_name, delay) in delayed_feedback:
                # Resolve aliases
                in_times = None
                in_values = None
                out_times = None
                out_values = None
                for variable in itertools.chain(self.differentiated_states, self.algebraic_states, self.controls):
                    for alias in self.variable_aliases(variable):
                        if alias.name == in_variable_name:
                            in_times = self.times(variable)
                            in_values = alias.sign * self.variable_nominal(
                                variable) * self.state_vector(variable, ensemble_member=ensemble_member)
                        if alias.name == out_variable_name:
                            out_times = self.times(variable)
                            out_values = self.variable_nominal(
                                variable) * self.state_vector(variable, ensemble_member=ensemble_member)
                            history_found = False
                            for history_alias in self.variable_aliases(variable):
                                if history_alias.name in history:
                                    out_times = np.concatenate(
                                        [history[history_alias.name].times[:-1], out_times])
                                    out_values = vertcat(
                                        [history[history_alias.name].values[:-1], out_values])
                                    history_found = True
                                    break
                            if not history_found:
                                logger.warning("No history available for delayed variable {}. Extrapolating t0 value backwards in time.".format(out_variable_name))
                            out_values *= alias.sign
                    if in_times is not None and out_times is not None:
                        break
                for variable in self.dae_variables['constant_inputs']:
                    variable = variable.getName()
                    for alias in self.variable_aliases(variable):
                        if alias.name == out_variable_name:
                            out_times = collocation_times
                            out_values = alias.sign * \
                                constant_inputs[variable]
                    if out_times is not None:
                        break
                if in_times is None:
                    raise Exception(
                        "Could not find variable with name {}".format(in_variable_name))
                if out_times is None:
                    raise Exception(
                        "Could not find variable with name {}".format(out_variable_name))

                # Set up delay constraints
                if len(collocation_times) != len(in_times):
                    x_in = interp1d(in_times, in_values,
                                    collocation_times, self.equidistant)
                else:
                    x_in = in_values
                x_out_delayed = interp1d(
                    out_times, out_values, collocation_times - delay, self.equidistant)

                g.append(x_in - x_out_delayed)
                lbg.extend(n_collocation_times * [0.0])
                ubg.extend(n_collocation_times * [0.0])

            # Objective
            f_member = self.objective(ensemble_member)
            if path_objective.size1() > 0:
                initial_path_objective = path_objective_function([parameters, 
                                                                  vertcat([initial_state
                                                                              , initial_derivatives
                                                                              , initial_constant_inputs,
                                                                              0.0,
                                                                              initial_path_variables]),
                                                                  X], False, True)
                f_member += initial_path_objective[0] + sumRows(discretized_path_objective)
            f.append(self.ensemble_member_probability(ensemble_member) * f_member)

            if logger.getEffectiveLevel() == logging.DEBUG:
                logger.debug(
                    "Adding objective {}".format(f_member))

            # Constraints
            constraints = self.constraints(ensemble_member)
            if logger.getEffectiveLevel() == logging.DEBUG:
                for constraint in constraints:
                    logger.debug(
                        "Adding constraint {}, {}, {}".format(*constraint))

            g_constraint = [f_constraint for (f_constraint, lb, ub) in constraints]
            g.extend(g_constraint)

            lbg_constraint = [lb for (f_constraint, lb, ub) in constraints]
            lbg.extend(lbg_constraint)
            
            ubg_constraint = [ub for (f_constraint, lb, ub) in constraints]
            ubg.extend(ubg_constraint)

            # Path constraints
            # We need to call self.path_constraints() again here, as the bounds may change from ensemble member to member.
            path_constraints = self.path_constraints(ensemble_member)
            if len(path_constraints) > 0:
                # We need to evaluate the path constraints at t0, as the initial time is not included in the accumulation.
                [initial_path_constraints] = path_constraints_function([parameters,
                                                                      vertcat([initial_state
                                                                              , initial_derivatives
                                                                              , initial_constant_inputs,
                                                                              0.0,
                                                                              initial_path_variables]),
                                                                      X], False, True)
                g.append(initial_path_constraints)
                g.append(discretized_path_constraints)

                lbg_path_constraints = np.empty(
                    (len(path_constraints), n_collocation_times))
                ubg_path_constraints = np.empty(
                    (len(path_constraints), n_collocation_times))
                for j, path_constraint in enumerate(path_constraints):
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        logger.debug(
                            "Adding path constraint {}, {}, {}".format(*path_constraint))

                    lb = path_constraint[1]
                    if isinstance(lb, Timeseries):
                        lb = self.interpolate(
                            collocation_times, lb.times, lb.values, -np.inf, -np.inf)

                    ub = path_constraint[2]
                    if isinstance(ub, Timeseries):
                        ub = self.interpolate(
                            collocation_times, ub.times, ub.values, np.inf, np.inf)

                    lbg_path_constraints[j, :] = lb
                    ubg_path_constraints[j, :] = ub
                lbg.extend(lbg_path_constraints.transpose().ravel())
                ubg.extend(ubg_path_constraints.transpose().ravel())

        # NLP function
        logger.info("Creating NLP function")

        # , {'jit': True, 'compiler': 'shell'})
        nlp = MXFunction('nlp', nlpIn(x=X), nlpOut(f=sumRows(vertcat(f)), g=vertcat(g)))

        # Done
        logger.info("Done transcribing problem")

        return discrete, lbx, ubx, lbg, ubg, x0, nlp

    def extract_results(self, ensemble_member=0):
        logger.info("Extracting results")

        # Gather results in a dictionary
        control_results = self.extract_controls(ensemble_member)
        state_results = self.extract_states(ensemble_member)

        # Merge dictionaries
        results = control_results
        results.update(state_results)

        logger.info("Done extracting results")

        # Return results dictionary
        return results

    @property
    def solver_input(self):
        return self._solver_input

    def solver_options(self):
        options = super(CollocatedIntegratedOptimizationProblem,
                        self).solver_options()
        if self._affine_collocation_constraints:
            options['jac_c_constant'] = 'yes'
        return options

    def integrator_options(self):
        """
        Configures the implicit function used for time step integration.

        :returns: A dictionary of CasADi :class:`ImplicitFunction` options.  See the CasADi documentation for details.
        """
        return {'linear_solver': 'csparse'}

    @property
    def controls(self):
        return self._controls

    def discretize_controls(self):
        # Default implementation: One single set of control inputs for all
        # ensembles
        count = 0
        for variable in self.controls:
            times = self.times(variable)
            n_times = len(times)

            count += n_times

        bounds = self.bounds()

        # We assume the seed for the controls to be identical for the entire ensemble.
        # After all, we don't use a stochastic tree if we end up here.
        seed = self.seed(ensemble_member=0)

        discrete = np.zeros(count, dtype=np.bool)

        lbx = -np.inf * np.ones(count)
        ubx = np.inf * np.ones(count)

        x0 = np.zeros(count)

        offset = 0
        for variable in self.controls:
            times = self.times(variable)
            n_times = len(times)

            discrete[offset:offset +
                     n_times] = self.variable_is_discrete(variable)

            for alias in self.variable_aliases(variable):
                try:
                    bound = bounds[alias.name]
                except KeyError:
                    continue

                nominal = self.variable_nominal(variable)
                if bound[0] != None:
                    if isinstance(bound[0], Timeseries):
                        lbx[offset:offset + n_times] = self.interpolate(
                            times, bound[0].times, bound[0].values, -np.inf, -np.inf) / nominal
                    else:
                        lbx[offset:offset + n_times] = bound[0] / nominal
                if bound[1] != None:
                    if isinstance(bound[1], Timeseries):
                        ubx[offset:offset + n_times] = self.interpolate(
                            times, bound[1].times, bound[1].values, +np.inf, +np.inf) / nominal
                    else:
                        ubx[offset:offset + n_times] = bound[1] / nominal

                try:
                    seed_k = seed[variable]
                    x0[offset:offset + n_times] = self.interpolate(
                        times, seed_k.times, seed_k.values, 0, 0) / nominal
                except KeyError:
                    pass

                break

            offset += n_times

        # Return number of control variables
        return count, discrete, lbx, ubx, x0

    def extract_controls(self, ensemble_member=0):
        # Solver output
        X = self.solver_output

        # Extract control inputs
        results = {}
        offset = 0
        for variable in self.controls:
            n_times = len(self.times(variable))
            results[variable] = np.array(self.variable_nominal(
                variable) * X[offset:offset + n_times, 0]).ravel()
            offset += n_times

            for alias in self.variable_aliases(variable):
                if alias.name != variable:
                    results[alias.name] = alias.sign * results[variable]

        # Done
        return results

    def control_vector(self, variable, ensemble_member=0):
        # Default implementation: One single set of control inputs for all
        # ensembles
        t0 = self.initial_time
        X = self.solver_input

        # Return array of indexes for control
        offset = 0
        for control_input in self.controls:
            times = self.times(control_input)
            n_times = len(times)
            if control_input == variable:
                return X[offset:offset + n_times]
            offset += n_times

        raise KeyError(variable)

    def control_at(self, variable, t, ensemble_member=0, scaled=False, extrapolate=True):
        # Default implementation: One single set of control inputs for all
        # ensembles
        t0 = self.initial_time
        X = self.solver_input

        offset = 0
        for control_input in self.controls:
            times = self.times(control_input)
            for alias in self.variable_aliases(control_input):
                if alias.name == variable:
                    nominal = self.variable_nominal(control_input)
                    n_times = len(times)
                    variable_values = X[offset:offset + n_times]
                    f_left, f_right = np.nan, np.nan
                    if t < t0:
                        history = self.history(ensemble_member)
                        history_found = False
                        for history_alias in self.variable_aliases(control_input):
                            if history_alias.name in history:
                                history_timeseries = history[
                                    history_alias.name]
                                if extrapolate:
                                    f_left = history_timeseries.values[0]
                                    f_right = history_timeseries.values[-1]
                                sym = history_alias.sign * \
                                    self.interpolate(
                                        t, history_timeseries.times, history_timeseries.values, f_left, f_right)
                                history_found = True
                        if not history_found:
                            if extrapolate:
                                sym = variable_values[0]
                            else:
                                sym = np.nan
                            if not scaled and nominal != 1:
                                sym *= nominal
                    else:
                        if extrapolate:
                            f_left = variable_values[0]
                            f_right = variable_values[-1]
                        sym = self.interpolate(
                            t, times, variable_values, f_left, f_right)
                        if not scaled and nominal != 1:
                            sym *= nominal
                    if alias.sign < 0:
                        sym *= -1
                    return sym
            offset += len(times)

        raise KeyError(variable)

    @property
    def differentiated_states(self):
        return self._differentiated_states

    @property
    def algebraic_states(self):
        return self._algebraic_states

    def discretize_states(self):
        # Default implementation: States for all ensemble members
        ensemble_member_size = 0

        # Space for collocated states
        for variable in itertools.chain(self.differentiated_states, self.algebraic_states, self._path_variable_names):
            if variable in self.integrated_states:
                ensemble_member_size += 1  # Initial state
            else:
                ensemble_member_size += len(self.times(variable))

        # Space for extra variables
        ensemble_member_size += len(self.extra_variables)

        # Space for initial states and derivatives
        ensemble_member_size += len(self.dae_variables['derivatives'])

        # Total space requirement
        count = self.ensemble_size * ensemble_member_size

        # Allocate arrays
        discrete = np.zeros(count, dtype=np.bool)

        lbx = -np.inf * np.ones(count)
        ubx = np.inf * np.ones(count)

        x0 = np.zeros(count)

        # Types
        for ensemble_member in range(self.ensemble_size):
            offset = ensemble_member * ensemble_member_size
            for variable in itertools.chain(self.differentiated_states, self.algebraic_states, self._path_variable_names):
                if variable in self.integrated_states:
                    discrete[offset] = self.variable_is_discrete(variable)

                    offset += 1

                else:
                    times = self.times(variable)
                    n_times = len(times)

                    discrete[offset:offset +
                             n_times] = self.variable_is_discrete(variable)

                    offset += n_times

            for k in xrange(len(self.extra_variables)):
                discrete[
                    offset + k] = self.variable_is_discrete(self.extra_variables[k].getName())

        # Bounds, defaulting to +/- inf, if not set
        bounds = self.bounds()

        for ensemble_member in range(self.ensemble_size):
            offset = ensemble_member * ensemble_member_size
            for variable in itertools.chain(self.differentiated_states, self.algebraic_states, self._path_variable_names):
                if variable in self.integrated_states:
                    for alias in self.variable_aliases(variable):
                        try:
                            bound = bounds[alias.name]
                        except KeyError:
                            continue

                        nominal = self.variable_nominal(variable)
                        if bound[0] != None:
                            if isinstance(bound[0], Timeseries):
                                lbx[offset] = self.interpolate(self.initial_time, bound[0].times, bound[
                                                               0].values, -np.inf, -np.inf) / nominal
                            else:
                                lbx[offset] = bound[0] / nominal
                        if bound[1] != None:
                            if isinstance(bound[1], Timeseries):
                                ubx[offset] = self.interpolate(self.initial_time, bound[1].times, bound[
                                                               1].values, +np.inf, +np.inf) / nominal
                            else:
                                ubx[offset] = bound[1] / nominal

                        break

                    offset += 1

                else:
                    times = self.times(variable)
                    n_times = len(times)

                    for alias in self.variable_aliases(variable):
                        try:
                            bound = bounds[alias.name]
                        except KeyError:
                            continue

                        nominal = self.variable_nominal(variable)
                        if bound[0] != None:
                            if isinstance(bound[0], Timeseries):
                                lbx[offset:offset + n_times] = self.interpolate(
                                    times, bound[0].times, bound[0].values, -np.inf, -np.inf) / nominal
                            else:
                                lbx[offset:offset + n_times] = bound[0] / nominal
                        if bound[1] != None:
                            if isinstance(bound[1], Timeseries):
                                ubx[offset:offset + n_times] = self.interpolate(
                                    times, bound[1].times, bound[1].values, +np.inf, +np.inf) / nominal
                            else:
                                ubx[offset:offset + n_times] = bound[1] / nominal

                        break

                    offset += n_times

            for k in xrange(len(self.extra_variables)):
                try:
                    bound = bounds[self.extra_variables[k].getName()]
                except KeyError:
                    continue

                if bound[0] != None:
                    lbx[offset + k] = bound[0]
                if bound[1] != None:
                    ubx[offset + k] = bound[1]

            # Initial guess based on provided seeds, defaulting to zero if no
            # seed is given
            seed = self.seed(ensemble_member)

            offset = ensemble_member * ensemble_member_size
            for variable in itertools.chain(self.differentiated_states, self.algebraic_states, self._path_variable_names):
                if variable in self.integrated_states:
                    try:
                        seed_k = seed[variable]
                        nominal = self.variable_nominal(variable)
                        x0[offset] = self.interpolate(
                            self.initial_time, seed_k.times, seed_k.values, 0, 0) / nominal
                    except KeyError:
                        pass

                    offset += 1

                else:
                    times = self.times(variable)
                    n_times = len(times)

                    try:
                        seed_k = seed[variable]
                        nominal = self.variable_nominal(variable)
                        x0[offset:offset + n_times] = self.interpolate(
                            times, seed_k.times, seed_k.values, 0, 0) / nominal
                    except KeyError:
                        pass

                    offset += n_times

            for k in xrange(len(self.extra_variables)):
                try:
                    x0[offset + k] = seed[self.extra_variables[k].getName()]
                except KeyError:
                    pass

        # Return number of state variables
        return count, discrete, lbx, ubx, x0

    def extract_states(self, ensemble_member=0):
        # Solver output
        X = self.solver_output

        # Discretization parameters
        control_size = self._control_size
        ensemble_member_size = self._state_size / self.ensemble_size

        # Extract control inputs
        results = {}

        # Perform integration, in order to extract integrated variables
        # We bundle all integrations into a single Function, so that subexpressions
        # are evaluated only once.
        if len(self.integrated_states) > 0:
            # Use integrators_mx to facilicate common subexpression
            # elimination.
            f = MXFunction('f', [self.solver_input], [
                           vertcat(self.integrators_mx)])
            f.setInput(X[0:])
            f.evaluate()
            integrators_output = f.getOutput()
            j = 0
            for variable in self.integrated_states:
                n = self.integrators[variable].size1()
                results[variable] = self.variable_nominal(
                    variable) * np.array(integrators_output[j:j + n, 0]).ravel()
                j += n

        # Extract collocated variables
        offset = control_size + ensemble_member * ensemble_member_size
        for variable in itertools.chain(self.differentiated_states, self.algebraic_states):
            if variable in self.integrated_states:
                offset += 1
            else:
                n_times = len(self.times(variable))
                results[variable] = np.array(self.variable_nominal(
                    variable) * X[offset:offset + n_times, 0]).ravel()
                offset += n_times

                for alias in self.variable_aliases(variable):
                    if alias.name != variable:
                        results[alias.name] = alias.sign * results[variable]

        # Extract path variables
        n_collocation_times = len(self.times())
        for variable in self.path_variables:
            variable = variable.getName()
            results[variable] = np.array(
                X[offset:offset + n_collocation_times, 0]).ravel()
            offset += n_collocation_times

        # Extract extra variables
        for k in xrange(len(self.extra_variables)):
            variable = self.extra_variables[k].getName()
            results[variable] = np.array(X[offset + k, 0]).ravel()

        # Done
        return results

    def state_vector(self, variable, ensemble_member=0):
        # Look up transcribe_problem() state.
        X = self.solver_input
        control_size = self._control_size
        ensemble_member_size = self._state_size / self.ensemble_size

        # Return array of indexes for state
        offset = control_size + ensemble_member * ensemble_member_size
        for free_variable in itertools.chain(self.differentiated_states, self.algebraic_states, self._path_variable_names):
            times = self.times(free_variable)
            n_times = len(times)
            if free_variable == variable:
                if free_variable in self.integrated_states:
                    return X[offset]
                else:
                    return X[offset:offset + n_times]
            if free_variable in self.integrated_states:
                offset += 1
            else:
                offset += n_times

        # Could not find state.  Try controls.
        return self.control_vector(variable, ensemble_member=ensemble_member)

    def state_at(self, variable, t, ensemble_member=0, scaled=False, extrapolate=True):
        if isinstance(variable, MX):
            variable = variable.getName()
        name = "{}[{},{}]{}".format(
            variable, ensemble_member, t - self.initial_time, 'S' if scaled else '')
        if extrapolate:
            name += 'E'
        try:
            return self._symbol_cache[name]
        except KeyError:
            # Look up transcribe_problem() state.
            t0 = self.initial_time
            X = self.solver_input
            control_size = self._control_size
            ensemble_member_size = self._state_size / self.ensemble_size

            # Fetch appropriate symbol, or value.
            found = False
            if not found:
                offset = control_size + ensemble_member * ensemble_member_size
                for free_variable in itertools.chain(self.differentiated_states, self.algebraic_states):
                    for alias in self.variable_aliases(free_variable):
                        if alias.name == variable:
                            times = self.times(free_variable)
                            n_times = len(times)
                            if free_variable in self.integrated_states:
                                nominal = 1
                                if t == self.initial_time:
                                    sym = alias.sign * X[offset]
                                    found = True
                                    break
                                else:
                                    variable_values = self.integrators[
                                        free_variable]
                            else:
                                nominal = self.variable_nominal(free_variable)
                                variable_values = X[offset:offset + n_times]
                            f_left, f_right = np.nan, np.nan
                            if t < t0:
                                history = self.history(ensemble_member)
                                history_found = False
                                for history_alias in self.variable_aliases(free_variable):
                                    if history_alias.name in history:
                                        history_timeseries = history[
                                            history_alias.name]
                                        if extrapolate:
                                            f_left = history_timeseries.values[
                                                0]
                                            f_right = history_timeseries.values[
                                                -1]
                                        sym = history_alias.sign * \
                                            self.interpolate(
                                                t, history_timeseries.times, history_timeseries.values, f_left, f_right)
                                        history_found = True
                                if not history_found:
                                    if extrapolate:
                                        sym = variable_values[0]
                                    else:
                                        sym = np.nan
                                    if not scaled and nominal != 1:
                                        sym *= nominal
                            else:
                                if extrapolate:
                                    f_left = variable_values[0]
                                    f_right = variable_values[-1]
                                sym = self.interpolate(
                                    t, times, variable_values, f_left, f_right)
                                if not scaled and nominal != 1:
                                    sym *= nominal
                            if alias.sign < 0:
                                sym *= -1
                            found = True
                            break
                    if found:
                        break
                    if free_variable in self.integrated_states:
                        offset += 1
                    else:
                        offset += len(self.times(free_variable))
            if not found:
                try:
                    sym = self.control_at(
                        variable, t, ensemble_member=ensemble_member, extrapolate=extrapolate)
                    found = True
                except KeyError:
                    pass
            if not found:
                constant_inputs = self.constant_inputs(ensemble_member)
                for constant_input_variable in constant_inputs.keys():
                    for alias in self.variable_aliases(constant_input_variable):
                        if alias.name == variable:
                            times = self.times(constant_input_variable)
                            n_times = len(times)
                            constant_input = constant_inputs[
                                constant_input_variable]
                            f_left, f_right = np.nan, np.nan
                            if extrapolate:
                                f_left = constant_input.values[0]
                                f_right = constant_input.values[-1]
                            sym = self.interpolate(
                                t, constant_input.times, constant_input.values, f_left, f_right)
                            sym *= alias.sign
                            found = True
                            break
                    if found:
                        break
            if not found:
                parameters = self.parameters(ensemble_member)
                for parameter in parameters.keys():
                    for alias in self.variable_aliases(parameter):
                        if alias.name == variable:
                            sym = parameters[parameter]
                            sym *= alias.sign
                            found = True
                            break
                    if found:
                        break
            if not found:
                raise KeyError(variable)

            # Cache symbol.
            self._symbol_cache[name] = sym

            return sym

    def extra_variable(self, extra_variable, ensemble_member=0):
        # Look up transcribe_problem() state.
        X = self.solver_input
        control_size = self._control_size
        ensemble_member_size = self._state_size / self.ensemble_size

        # Compute position in state vector
        offset = control_size + ensemble_member * ensemble_member_size
        for variable in itertools.chain(self.differentiated_states, self.algebraic_states):
            if variable in self.integrated_states:
                offset += 1
            else:
                n_times = len(self.times(variable))
                offset += n_times

        n_collocation_times = len(self.times())
        for variable in self.path_variables:
            offset += n_collocation_times

        for k in xrange(len(self.extra_variables)):
            variable = self.extra_variables[k].getName()
            if variable == extra_variable:
                return X[offset + k]

        raise KeyError(variable)

    def states_in(self, variable, t0=None, tf=None, ensemble_member=0):
        # Time stamps for this variale
        times = self.times(variable)

        # Set default values
        if t0 is None:
            t0 = times[0]
        if tf is None:
            tf = times[-1]

        # Compute combined points
        if t0 < times[0]:
            history = self.history(ensemble_member)
            try:
                htimes = history[variable].times[:-1]
            except KeyError:
                htimes = []
            history_and_times = np.hstack((htimes, times))
        else:
            history_and_times = times

        # Yield left boundary point
        yield self.state_at(variable, t0, ensemble_member=ensemble_member)

        # Yield interior points
        for point in history_and_times:
            if point > t0 and point < tf:
                yield self.state_at(variable, point, ensemble_member=ensemble_member)
            if point >= tf:
                break

        # Yield right boundary point, if it differs from the left boundary
        # point
        if t0 != tf:
            yield self.state_at(variable, tf, ensemble_member=ensemble_member)

    def integral(self, variable, t0=None, tf=None, ensemble_member=0):
        # Time stamps for this variale
        times = self.times(variable)

        # Set default values
        if t0 is None:
            t0 = times[0]
        if tf is None:
            tf = times[-1]

        # Compute combined points
        if t0 < times[0]:
            history = self.history(ensemble_member)
            try:
                htimes = history[variable].times[:-1]
            except KeyError:
                htimes = []
            history_and_times = np.hstack((htimes, times))
        else:
            history_and_times = times

        # Collect time stamps and states, "knots".
        t = []
        x = []
        t.append(t0)
        x.append(self.state_at(variable, t0, ensemble_member=ensemble_member))
        for point in history_and_times:
            if point > t0 and point < tf:
                t.append(point)
                x.append(self.state_at(variable, point,
                                       ensemble_member=ensemble_member))
            if point >= tf:
                break
        if t0 != tf:
            t.append(tf)
            x.append(self.state_at(variable, tf,
                                   ensemble_member=ensemble_member))
        t = vertcat(t)
        x = vertcat(x)

        # Integrate knots using trapezoid rule
        x_avg = 0.5 * (x[:x.size1() - 1] + x[1:])
        dt = t[1:] - t[:x.size1() - 1]
        return sumRows(x_avg * dt)

    def der(self, variable):
        # Look up the derivative variable for the given non-derivative variable
        for i, var in enumerate(self.differentiated_states):
            for alias in self.variable_aliases(var):
                if alias.name == variable:
                    return alias.sign * self.dae_variables['derivatives'][i]
        for i, var in enumerate(itertools.chain(self.algebraic_states, self.controls)):
            for alias in self.variable_aliases(var):
                if alias.name == variable:
                    return alias.sign * self._algebraic_and_control_derivatives[i]
        raise KeyError

    def der_at(self, variable, t, ensemble_member=0):
        # Special case t being t0 for differentiated states
        if t == self.initial_time:
            # We have a special symbol for t0 derivatives
            X = self.solver_input
            control_size = self._control_size
            ensemble_member_size = self._state_size / self.ensemble_size

            for i, state in enumerate(self.differentiated_states):
                for alias in self.variable_aliases(state):
                    if alias.name == variable:
                        return alias.sign * X[control_size + (ensemble_member + 1) * ensemble_member_size - len(self.dae_variables['derivatives']) + i]
            # Fall through, in case 'variable' is not a differentiated state.

        # Time stamps for this variale
        times = self.times(variable)

        if t <= self.initial_time:
            # Derivative requested for t0 or earlier.  We need the history.
            history = self.history(ensemble_member)
            try:
                htimes = history[variable].times[:-1]
            except KeyError:
                htimes = []
            history_and_times = np.hstack((htimes, times))
        else:
            history_and_times = times

        # Special case t being the initial available point.  In this case, we have
        # no derivative information available.
        if t == history_and_times[0]:
            return 0.0

        # Handle t being an interior point, or t0 for a non-differentiated
        # state
        for i in range(len(history_and_times)):
            # Use finite differences when between collocation points, and
            # backward finite differences when on one.
            if t > history_and_times[i] and t <= history_and_times[i + 1]:
                dx = self.state_at(variable, history_and_times[i + 1], ensemble_member=ensemble_member) - self.state_at(
                    variable, history_and_times[i], ensemble_member=ensemble_member)
                dt = history_and_times[i + 1] - history_and_times[i]
                return dx / dt

        # t does not belong to any collocation point interval
        raise IndexError

    def map_path_expression(self, expr, ensemble_member):
        # Expression as function of states and derivatives
        states = self.dae_variables['states'] + self.dae_variables['algebraics'] + self.dae_variables['control_inputs']
        states_and_path_variables = states + self.path_variables
        derivatives = self.dae_variables['derivatives'] + self._algebraic_and_control_derivatives

        f = MXFunction('f', [vertcat(states_and_path_variables), vertcat(derivatives),
            vertcat(self.dae_variables['constant_inputs']), vertcat(self.dae_variables['parameters']),
            self.dae_variables['time'][0], self.solver_input], [expr])
        fmap = f.map('fmap', len(self.times()))

        # Discretization settings
        collocation_times = self.times()
        n_collocation_times = len(collocation_times)
        dt = transpose(collocation_times[1:] - collocation_times[:-1])
        t0 = self.initial_time

        # Prepare interpolated state and path variable vectors
        accumulation_states = [None] * len(states_and_path_variables)
        for i, state in enumerate(states_and_path_variables):
            state = state.getName()
            times = self.times()
            values = self.state_vector(state, ensemble_member)
            if len(times) != n_collocation_times:
                accumulation_states[i] = interp1d(times, values, collocation_times)
            else:
                accumulation_states[i] = values
            nominal = self.variable_nominal(state)
            if nominal != 1:
                accumulation_states[i] *= nominal
        accumulation_states = transpose(horzcat(accumulation_states))

        # Prepare derivatives (backwards differencing, consistent with the evaluation of path expressions during transcription)
        accumulation_derivatives = [None] * len(derivatives)
        for i, state in enumerate(states):
            state = state.getName()
            accumulation_derivatives[i] = horzcat([self.der_at(state, t0, ensemble_member),
                (accumulation_states[i, 1:] - accumulation_states[i, :-1]) / dt])
        accumulation_derivatives = vertcat(accumulation_derivatives)

        # Prepare parameters
        parameters = self.parameters(ensemble_member)
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

        # Prepare constant inputs
        constant_inputs = self.constant_inputs(ensemble_member)
        accumulation_constant_inputs = [None] * len(self.dae_variables['constant_inputs'])
        for i, variable in enumerate(self.dae_variables['constant_inputs']):
            found = False
            for alias in self.variable_aliases(variable.getName()):
                if alias.name in constant_inputs:
                    constant_input = constant_inputs[alias.name]
                    values = constant_input.values
                    if isinstance(values, MX):
                        [values] = substitute([values], self.dae_variables['parameters'], parameter_values)
                    elif np.any([not MX(value).isConstant() for value in values]):
                        values = substitute(values, self.dae_variables['parameters'], parameter_values)
                    accumulation_constant_inputs[i] = alias.sign * self.interpolate(
                        collocation_times, constant_input.times, values, 0.0, 0.0)
                    found = True
                    break
            if not found:
                raise Exception("No data specified for constant input {}".format(variable.getName()))
        accumulation_constant_inputs = transpose(horzcat(accumulation_constant_inputs))

        # Map
        [values] = fmap([accumulation_states, accumulation_derivatives,
            accumulation_constant_inputs, repmat(vertcat(parameter_values), 1, n_collocation_times),
            np.transpose(collocation_times), repmat(self.solver_input, 1, n_collocation_times)])
        return transpose(values)
