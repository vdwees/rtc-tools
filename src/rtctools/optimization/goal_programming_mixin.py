from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Callable, Dict, List, Union
import casadi as ca
import numpy as np
import itertools
import logging
import sys

from rtctools._internal.alias_tools import AliasDict

from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


class Goal(metaclass = ABCMeta):
    """
    Base class for lexicographic goal programming goals.

    A goal is defined by overriding the :func:`function` method, and setting at least the
    ``function_range`` class variable.

    :cvar function_range:   Range of goal function.  *Required*.
    :cvar function_nominal: Nominal value of function. Used for scaling.  Default is ``1``.
    :cvar target_min:       Desired lower bound for goal function.  Default is ``numpy.nan``.
    :cvar target_max:       Desired upper bound for goal function.  Default is ``numpy.nan``.
    :cvar priority:         Integer priority of goal.  Default is ``1``.
    :cvar weight:           Optional weighting applied to the goal.  Default is ``1.0``.
    :cvar order:            Penalization order of goal violation.  Default is ``2``.
    :cvar critical:         If ``True``, the algorithm will abort if this goal cannot be fully met.  Default is ``False``.

    The target bounds indicate the range within the function should stay, *if possible*.  Goals
    are, in that sense, *soft*, as opposed to standard hard constraints.

    Four types of goals can be created:

    1. Minimization goal if no target bounds are set:

       .. math::

            \\min f

    2. Lower bound goal if ``target_min`` is set:

        .. math::

            m \\leq f

    3. Upper bound goal if ``target_max`` is set:

        .. math::

            f \\leq M

    4. Combined lower and upper bound goal if ``target_min`` and ``target_max`` are both set:

        .. math::

            m \\leq f \\leq M

    Lower priority goals take precedence over higher priority goals.

    Goals with the same priority are weighted off against each other in a
    single objective function.

    The goal violation value is taken to the order'th power in the objective function of the final
    optimization problem.

    Example definition of the point goal :math:`x(t) \geq 1.1` for :math:`t=1.0` at priority 1::

        class MyGoal(Goal):
            def function(self, optimization_problem, ensemble_member):
                # State 'x' at time t = 1.0
                t = 1.0
                return optimization_problem.state_at('x', t, ensemble_member)

            function_range = (1.0, 2.0)
            target_min = 1.1
            priority = 1

    Example definition of the path goal :math:`x(t) \geq 1.1` for all :math:`t` at priority 2::

        class MyPathGoal(Goal):
            def function(self, optimization_problem, ensemble_member):
                # State 'x' at any point in time
                return optimization_problem.state('x')

            function_range = (1.0, 2.0)
            target_min = 1.1
            priority = 2

    Note that for path goals, the ensemble member index is not passed to the call
    to :func:`OptimizationProblem.state`.  This call returns a time-independent symbol
    that is also independent of the active ensemble member.  Path goals are
    applied to all times and all ensemble members simultaneously.

    """

    @abstractmethod
    def function(self, optimization_problem: OptimizationProblem, ensemble_member: int) -> ca.MX:
        """
        This method returns a CasADi :class:`MX` object describing the goal function.

        :returns: A CasADi :class:`MX` object.
        """
        pass

    #: Range of goal function
    function_range = (np.nan, np.nan)

    #: Nominal value of function (used for scaling)
    function_nominal = 1.0

    #: Desired lower bound for goal function
    target_min = np.nan

    #: Desired upper bound for goal function
    target_max = np.nan

    #: Lower priority gols take precedence over higher priority goals.
    priority = 1

    #: Goals with the same priority are weighted off against each other in a
    # single objective function.
    weight = 1.0

    #: The goal violation value is taken to the order'th power in the objective
    # function.
    order = 2

    #: Critical goals must always be fully satisfied.
    critical = False

    #: Absolute relaxation applied to the optimized values of this goal
    relaxation = 0.0

    #: Timeseries ID for function value data (optional)
    function_value_timeseries_id = None

    #: Timeseries ID for goal violation data (optional)
    violation_timeseries_id = None

    @property
    def has_target_min(self) -> bool:
        """
        ``True`` if the user goal has min bounds.
        """
        if isinstance(self.target_min, Timeseries):
            return True
        else:
            return np.isfinite(self.target_min)

    @property
    def has_target_max(self) -> bool:
        """
        ``True`` if the user goal has max bounds.
        """
        if isinstance(self.target_max, Timeseries):
            return True
        else:
            return np.isfinite(self.target_max)

    @property
    def has_target_bounds(self) -> bool:
        """
        ``True`` if the user goal has min/max bounds.
        """
        return (self.has_target_min or self.has_target_max)

    @property
    def is_empty(self) -> bool:
        min_empty = (isinstance(self.target_min, Timeseries) and not np.any(np.isfinite(self.target_min.values)))
        max_empty = (isinstance(self.target_max, Timeseries) and not np.any(np.isfinite(self.target_max.values)))
        return min_empty and max_empty

    def get_function_key(self, optimization_problem: OptimizationProblem, ensemble_member: int) -> str:
        """
        Returns a key string uniquely identifying the goal function.  This
        is used to eliminate linearly dependent constraints from the optimization problem.
        """
        if hasattr(self, 'function_key'):
            return self.function_key

        # This must be deterministic.  See RTCTOOLS-485.
        if not hasattr(Goal, '_function_key_counter'):
            Goal._function_key_counter = 0
        self.function_key = '{}_{}'.format(self.__class__.__name__, Goal._function_key_counter)
        Goal._function_key_counter += 1

        return self.function_key

    def __repr__(self) -> str:
        return '{}(priority={}, target_min={}, target_max={})'.format(self.__class__, self.priority, self.target_min, self.target_max)


class StateGoal(Goal, metaclass = ABCMeta):
    """
    Base class for lexicographic goal programming path goals that act on a single model state.

    A state goal is defined by setting at least the ``state`` class variable.

    :cvar state:            State on which the goal acts.  *Required*.
    :cvar target_min:       Desired lower bound for goal function.  Default is ``numpy.nan``.
    :cvar target_max:       Desired upper bound for goal function.  Default is ``numpy.nan``.
    :cvar priority:         Integer priority of goal.  Default is ``1``.
    :cvar weight:           Optional weighting applied to the goal.  Default is ``1.0``.
    :cvar order:            Penalization order of goal violation.  Default is ``2``.
    :cvar critical:         If ``True``, the algorithm will abort if this goal cannot be fully met.  Default is ``False``.

    Example definition of the goal :math:`x(t) \geq 1.1` for all :math:`t` at priority 2::

        class MyStateGoal(StateGoal):
            state = 'x'
            target_min = 1.1
            priority = 2

    Contrary to ordinary ``Goal`` objects, ``PathGoal`` objects need to be initialized with an
    ``OptimizationProblem`` instance to allow extraction of state metadata, such as bounds and
    nominal values.  Consequently, state goals must be instantiated as follows::

        my_state_goal = MyStateGoal(optimization_problem)

    Note that ``StateGoal`` is a helper class.  State goals can also be defined using ``Goal`` as direct base class,
    by implementing the ``function`` method and providing the ``function_range`` and ``function_nominal``
    class variables manually.

    """

    #: The state on which the goal acts.
    state = None

    def __init__(self, optimization_problem):
        """
        Initialize the state goal object.

        :param optimization_problem: ``OptimizationProblem`` instance.
        """

        # Check whether a state has been specified
        if self.state is None:
            raise Exception('Please specify a state.')

        # Extract state range from model
        try:
            self.function_range = optimization_problem.bounds()[self.state]
        except KeyError:
            raise Exception('State {} has no bounds or does not exist in the model.'.format(self.state))

        if self.function_range[0] is None:
            raise Exception('Please provide a lower bound for state {}.'.format(self.state))
        if self.function_range[1] is None:
            raise Exception('Please provide an upper bound for state {}.'.format(self.state))

        # Extract state nominal from model
        self.function_nominal = optimization_problem.variable_nominal(self.state)

        # Set function key
        self.function_key = self.state

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)

    def __repr__(self):
        return '{}(priority={}, state={}, target_min={}, target_max={})'.format(self.__class__, self.priority, self.state, self.target_min, self.target_max)


class GoalProgrammingMixin(OptimizationProblem, metaclass = ABCMeta):
    """
    Adds lexicographic goal programming to your optimization problem.
    """

    class __GoalConstraint:

        def __init__(self, goal: Goal, function: Callable[[OptimizationProblem], ca.MX], m: Union[float, Timeseries], M: Union[float, Timeseries], optimized: bool):
            self.goal = goal
            self.function = function
            self.min = m
            self.max = M
            self.optimized = optimized

    def __init__(self, **kwargs):
        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

        # Initialize empty lists, so that the overridden methods may be called outside of the goal programming loop,
        # for example in pre().
        self.__subproblem_epsilons = []
        self.__subproblem_path_epsilons = []
        self.__subproblem_path_timeseries = []
        self.__subproblem_objectives = []
        self.__subproblem_constraints = []
        self.__subproblem_path_constraints = []

        self.__original_constant_input_keys = {}

    @property
    def extra_variables(self):
        return self.__subproblem_epsilons

    @property
    def path_variables(self):
        return self.__subproblem_path_epsilons.copy()

    def bounds(self):
        bounds = super().bounds()
        for epsilon in self.__subproblem_epsilons + self.__subproblem_path_epsilons:
            bounds[epsilon.name()] = (0.0, 1.0)
        return bounds

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)

        if not ensemble_member in self.__original_constant_input_keys:
            self.__original_constant_input_keys[ensemble_member] = set(constant_inputs.keys())

        # Remove min/max timeseries of previous priorities
        for k in set(constant_inputs.keys()):
            if k not in self.__original_constant_input_keys[ensemble_member]:
                del constant_inputs[k]

        # Append min/max timeseries to the constant inputs. Note that min/max
        # timeseries are shared between all ensemble members.
        for (variable, value) in self.__subproblem_path_timeseries:
            if not isinstance(value, Timeseries):
                value = Timeseries(self.times(), np.full_like(self.times(), value))
            constant_inputs[variable] = value
        return constant_inputs

    def seed(self, ensemble_member):
        if self.__first_run:
            seed = super().seed(ensemble_member)
        else:
            # Seed with previous results
            seed = AliasDict(self.alias_relation)
            for key, result in self.__results[ensemble_member].items():
                times = self.times(key)
                if len(result) == len(times):
                    # Only include seed timeseries which are consistent
                    # with the specified time stamps.
                    seed[key] = Timeseries(times, result)

        # Seed epsilons
        for epsilon in self.__subproblem_epsilons:
            seed[epsilon.name()] = 1.0

        times = self.times()
        for epsilon in self.__subproblem_path_epsilons:
            seed[epsilon.name()] = Timeseries(times, np.ones(len(times)))

        return seed

    def objective(self, ensemble_member):
        if len(self.__subproblem_objectives) > 0:
            acc_objective = ca.sum1(ca.vertcat(*[o(self, ensemble_member) for o in self.__subproblem_objectives]))
            return acc_objective / len(self.__subproblem_objectives)
        else:
            return ca.MX(0)

    def path_objective(self, ensemble_member):
        if len(self.__subproblem_path_objectives) > 0:
            acc_objective = ca.sum1(ca.vertcat(*[o(self, ensemble_member) for o in self.__subproblem_path_objectives]))
            return acc_objective / len(self.__subproblem_path_objectives)
        else:
            return ca.MX(0)

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)
        for l in self.__subproblem_constraints[ensemble_member].values():
            constraints.extend(((constraint.function(self), constraint.min, constraint.max) for constraint in l))
        return constraints

    def path_constraints(self, ensemble_member):
        path_constraints = super().path_constraints(ensemble_member)
        for l in self.__subproblem_path_constraints[ensemble_member].values():
            path_constraints.extend(((constraint.function(self), constraint.min, constraint.max) for constraint in l))
        return path_constraints

    def solver_options(self):
        # Call parent
        options = super().solver_options()

        # Make sure constant states, such as min/max timeseries for violation variables,
        # are turned into parameters for the final optimization problem.
        options['fixed_variable_treatment'] = 'make_parameter'

        if not self.goal_programming_options()['mu_reinit']:
            options['mu_strategy'] = 'monotone'
            options['gather_stats'] = True
            if not self.__first_run:
                options['mu_init'] = self.solver_stats['iterations'][
                    'mu'][-1]

        # Done
        return options

    def goal_programming_options(self) -> Dict[str, Union[float, bool]]:
        """
        Returns a dictionary of options controlling the goal programming process.

        +---------------------------+-----------+---------------+
        | Option                    | Type      | Default value |
        +===========================+===========+===============+
        | ``constraint_relaxation`` | ``float`` | ``0.0``       |
        +---------------------------+-----------+---------------+
        | ``mu_reinit``             | ``bool``  | ``True``      |
        +---------------------------+-----------+---------------+
        | ``fix_minimized_values``  | ``bool``  | ``True``      |
        +---------------------------+-----------+---------------+
        | ``check_monotonicity``    | ``bool``  | ``True``      |
        +---------------------------+-----------+---------------+
        | ``equality_threshold``    | ``float`` | ``1e-8``      |
        +---------------------------+-----------+---------------+
        | ``interior_distance``     | ``float`` | ``1e-6``      |
        +---------------------------+-----------+---------------+

        Constraints generated by the goal programming algorithm are relaxed by applying the specified relaxation.
        Use of this option is normally not required.

        A goal is considered to be violated if the violation, scaled between 0 and 1, is greater than the specified tolerance.
        Violated goals are fixed.  Use of this option is normally not required.

        The Ipopt barrier parameter ``mu`` is normally re-initialized a every iteration of the goal programming algorithm, unless
        mu_reinit is set to ``False``.  Use of this option is normally not required.

        If ``fix_minimized_values`` is set to ``True``, goal functions will be set to equal their
        optimized values in optimization problems generated during subsequent priorities.  Otherwise,
        only as upper bound will be set.  Use of this option is normally not required.

        If ``check_monotonicity`` is set to ``True``, then it will be checked whether goals with the same
        function key form a monotonically decreasing sequence with regards to the target interval.

        The option ``equality_threshold`` controls when a two-sided inequality constraint is folded into
        an equality constraint.

        The option ``interior_distance`` controls the distance from the scaled target bounds, starting
        from which the function value is considered to lie in the interior of the target space.

        :returns: A dictionary of goal programming options.
        """

        options = {}

        options['mu_reinit'] = True
        options['constraint_relaxation'] = 0.0  # Disable by default
        options['violation_tolerance'] = np.inf # Disable by default
        options['fix_minimized_values'] = True
        options['check_monotonicity'] = True
        options['equality_threshold'] = 1e-8
        options['interior_distance'] = 1e-6

        return options

    def goals(self) -> List[Goal]:
        """
        User problem returns list of :class:`Goal` objects.

        :returns: A list of goals.
        """
        return []

    def path_goals(self) -> List[Goal]:
        """
        User problem returns list of path :class:`Goal` objects.

        :returns: A list of path goals.
        """
        return []

    def __add_goal_constraint(self, goal, epsilon, ensemble_member, options):
        # Check existing constraints for this state.
        constraints = self.__subproblem_constraints[
            ensemble_member].get(goal.get_function_key(self, ensemble_member), [])
        for constraint in constraints:
            if constraint.goal == goal:
                continue
            if constraint.min == constraint.max:
                # This variable has already been fixed.  Don't add new
                # constraints for it.
                return
            elif constraint.goal.has_target_min:
                if goal.target_min < constraint.goal.target_min:
                    raise Exception(
                        "Target minimum of goal {} must be greater or equal than target minimum of goal {}.".format(goal, constraint.goal))
            elif constraint.goal.has_target_max:
                if goal.target_max > constraint.goal.target_max:
                    raise Exception(
                        "Target maximum of goal {} must be less or equal than target maximum of goal {}".format(goal, constraint.goal))

        # Check goal consistency
        if goal.has_target_min and goal.has_target_max:
            if goal.target_min > goal.target_max:
                raise Exception("Target minimum exceeds target maximum for goal {}".format(goal))

        if isinstance(epsilon, ca.MX):
            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation
                # variables epsilon bounded between 0 and 1.
                if goal.has_target_min:
                    constraint = self.__GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: (goal.function(problem, ensemble_member) - problem.extra_variable(
                        epsilon.name(), ensemble_member=ensemble_member) * (goal.function_range[0] - goal.target_min) - goal.target_min) / goal.function_nominal, 0.0, np.inf, False)
                    constraints.append(constraint)
                if goal.has_target_max:
                    constraint = self.__GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: (goal.function(problem, ensemble_member) - problem.extra_variable(
                        epsilon.name(), ensemble_member=ensemble_member) * (goal.function_range[1] - goal.target_max) - goal.target_max) / goal.function_nominal, -np.inf, 0.0, False)
                    constraints.append(constraint)

            # TODO forgetting max like this.
            # Epsilon is not fixed yet.  This constraint is therefore linearly independent of any existing constraints,
            # and we add it to the list of constraints for this state.  We keep the existing constraints to ensure
            # that the attainment of previous goals is not worsened.
            self.__subproblem_constraints[ensemble_member][
                goal.get_function_key(self, ensemble_member)] = constraints
        else:
            fix_value = False

            constraint = self.__GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal: goal.function(
                problem, ensemble_member) / goal.function_nominal, -np.inf, np.inf, True)
            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation
                # variables epsilon bounded between 0 and 1.
                if epsilon <= options['violation_tolerance']:
                    if goal.has_target_min:
                        constraint.min = (
                            epsilon * (goal.function_range[0] - goal.target_min) + goal.target_min - goal.relaxation) / goal.function_nominal
                    if goal.has_target_max:
                        constraint.max = (
                            epsilon * (goal.function_range[1] - goal.target_max) + goal.target_max + goal.relaxation) / goal.function_nominal
                    if goal.has_target_min and goal.has_target_max:
                        if abs(constraint.min - constraint.max) < options['equality_threshold']:
                            avg = 0.5 * (constraint.min + constraint.max)
                            constraint.min = constraint.max = avg
                else:
                    # Equality constraint to optimized value
                    fix_value = True

                    function = Function('function', [self.solver_input], [
                                          goal.function(self, ensemble_member)])
                    value = function(self.solver_output)

                    constraint.min = (value - goal.relaxation) / goal.function_nominal
                    constraint.max = (value + goal.relaxation) / goal.function_nominal
            else:
                # Epsilon encodes the position within the function range.
                fix_value = True

                constraint.min = (epsilon - goal.relaxation) / goal.function_nominal
                constraint.max = (epsilon + goal.relaxation) / goal.function_nominal

            # Epsilon is fixed.  Override previous {min,max} constraints for
            # this state.
            if not fix_value:
                for existing_constraint in constraints:
                    if goal is not existing_constraint.goal and existing_constraint.optimized:
                        constraint.min = max(constraint.min, existing_constraint.min)
                        constraint.max = min(constraint.max, existing_constraint.max)

            # Ensure consistency of bounds.  Bounds may become inconsistent due to
            # small numerical computation errors.
            constraint.min = min(constraint.min, constraint.max)

            self.__subproblem_constraints[ensemble_member][
                goal.get_function_key(self, ensemble_member)] = [constraint]

    def __add_path_goal_constraint(self, goal, epsilon, ensemble_member, options, min_series=None, max_series=None):
        # Generate list of min and max values
        times = self.times()

        def _min_max_arrays(g):
            m, M = None, None
            if isinstance(g.target_min, Timeseries):
                m = self.interpolate(
                    times, g.target_min.times, g.target_min.values, -np.inf, -np.inf)
            else:
                m = g.target_min * np.ones(len(times))
            if isinstance(g.target_max, Timeseries):
                M = self.interpolate(
                    times, g.target_max.times, g.target_max.values, np.inf, np.inf)
            else:
                M = g.target_max * np.ones(len(times))
            return m, M

        goal_m, goal_M = _min_max_arrays(goal)

        # Check existing constraints for this state.
        constraints = self.__subproblem_path_constraints[
            ensemble_member].get(goal.get_function_key(self, ensemble_member), [])
        for constraint in constraints:
            if constraint.goal == goal or not constraint.optimized:
                continue
            constraint_m, constraint_M = _min_max_arrays(constraint.goal)
            if np.all(constraint.min == constraint.max):
                # This variable has already been fixed.  Don't add new
                # constraints for it.
                return
            elif options['check_monotonicity']:
                if constraint.goal.has_target_min:
                    indices = np.where(np.logical_not(np.logical_or(
                        np.isnan(goal_m), np.isnan(constraint_m))))
                    if np.any(goal_m[indices] < constraint_m[indices]):
                        raise Exception(
                            "Minimum value of goal {} less than minimum of higher priority goal {}".format(goal, constraint.goal))
                elif constraint.goal.has_target_max:
                    indices = np.where(np.logical_not(np.logical_or(
                        np.isnan(goal_M), np.isnan(constraint_M))))
                    if np.any(goal_M[indices] > constraint_M[indices]):
                        raise Exception(
                            "Maximum value of goal {} greater than maximum of higher priority goal {}".format(goal, constraint.goal))

        # Check goal consistency
        if goal.has_target_min and goal.has_target_max:
            indices = np.where(np.logical_not(np.logical_or(
                np.isnan(goal_m), np.isnan(goal_M))))
            if np.any(goal_m[indices] > goal_M[indices]):
                raise Exception("Target minimum exceeds target maximum for goal {}".format(goal))

        if isinstance(epsilon, ca.MX):
            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation
                # variables epsilon bounded between 0 and 1.
                if goal.has_target_min:
                    constraint = self.__GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: ca.if_else(problem.variable(min_series) > -sys.float_info.max, (goal.function(problem, ensemble_member) - problem.variable(
                        epsilon.name()) * (goal.function_range[0] - problem.variable(min_series)) - problem.variable(min_series)) / goal.function_nominal, 0.0), 0.0, np.inf, False)
                    constraints.append(constraint)
                if goal.has_target_max:
                    constraint = self.__GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: ca.if_else(problem.variable(max_series) < sys.float_info.max, (goal.function(problem, ensemble_member) - problem.variable(
                        epsilon.name()) * (goal.function_range[1] - problem.variable(max_series)) - problem.variable(max_series)) / goal.function_nominal, 0.0), -np.inf, 0.0, False)
                    constraints.append(constraint)

            # TODO forgetting max like this.
            # Epsilon is not fixed yet.  This constraint is therefore linearly independent of any existing constraints,
            # and we add it to the list of constraints for this state.  We keep the existing constraints to ensure
            # that the attainment of previous goals is not worsened.
            self.__subproblem_path_constraints[ensemble_member][
                goal.get_function_key(self, ensemble_member)] = constraints
        else:
            fix_value = False

            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation
                # variables epsilon bounded between 0 and 1.
                m, M = np.full_like(times, -np.inf, dtype=np.float64), np.full_like(times, np.inf, dtype=np.float64)

                # Compute each min, max value separately for every time step
                for i, t in enumerate(times):
                    if np.isfinite(goal_m[i]) or np.isfinite(goal_M[i]):
                        if epsilon[i] <= options['violation_tolerance']:
                            if np.isfinite(goal_m[i]):
                                m[i] = (epsilon[i] * (goal.function_range[0] -
                                                      goal_m[i]) + goal_m[i] - goal.relaxation) / goal.function_nominal
                            if np.isfinite(goal_M[i]):
                                M[i] = (epsilon[i] * (goal.function_range[1] -
                                                      goal_M[i]) + goal_M[i] + goal.relaxation) / goal.function_nominal
                            if np.isfinite(goal_m[i]) and np.isfinite(goal_M[i]):
                                if abs(m[i] - M[i]) < options['equality_threshold']:
                                    avg = 0.5 * (m[i] + M[i])
                                    m[i] = M[i] = avg
                        else:
                            # Equality constraint to optimized value
                            # TODO this does not perform well.
                            variables = self.dae_variables['states'] + self.dae_variables[
                                'algebraics'] + self.dae_variables['control_inputs'] + self.dae_variables['constant_inputs']
                            values = [self.state_at(
                                variable, t, ensemble_member=ensemble_member) for variable in variables]
                            [function] = ca.substitute(
                                [goal.function(self, ensemble_member)], variables, values)
                            function = Function(
                                'function', [self.solver_input], [function])
                            value = function(self.solver_output)

                            m[i] = (value - goal.relaxation) / goal.function_nominal
                            M[i] = (value + goal.relaxation) / goal.function_nominal
            else:
                # Epsilon encodes the position within the function range.
                fix_value = True

                if options['fix_minimized_values']:
                    m = (epsilon - goal.relaxation) / goal.function_nominal
                else:
                    m = -np.inf * np.ones(len(times))
                M = (epsilon + goal.relaxation) / goal.function_nominal

            constraint = self.__GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal: goal.function(
                problem, ensemble_member) / goal.function_nominal, Timeseries(times, m), Timeseries(times, M), True)

            # Epsilon is fixed. Propagate/override previous {min,max}
            # constraints for this state.
            if not fix_value:
                for existing_constraint in constraints:
                    if goal is not existing_constraint.goal and existing_constraint.optimized:
                        constraint.min = Timeseries(times, np.maximum(constraint.min.values, existing_constraint.min.values))
                        constraint.max = Timeseries(times, np.minimum(constraint.max.values, existing_constraint.max.values))

            # Ensure consistency of bounds.  Bounds may become inconsistent due to
            # small numerical computation errors.
            constraint.min = Timeseries(times, np.minimum(constraint.min.values, constraint.max.values))

            self.__subproblem_path_constraints[ensemble_member][
                goal.get_function_key(self, ensemble_member)] = [constraint]

    def optimize(self, preprocessing=True, postprocessing=True, log_solver_failure_as_error=True):
        # Do pre-processing
        if preprocessing:
            self.pre()

        # Group goals into subproblems
        subproblems = []
        goals = self.goals()
        path_goals = self.path_goals()

        # Validate goal definitions
        for goal in itertools.chain(goals, path_goals):
            m, M = ca.MX(goal.function_range[0]), ca.MX(goal.function_range[1])

            if not m.is_regular() or not M.is_regular():
                raise Exception("No function range specified for goal {}".format(goal))

            if m >= M:
                raise Exception("Invalid function range for goal {}.".format(goal))

            if goal.function_nominal <= 0:
                raise Exception("Nonpositive nominal value specified for goal {}".format(goal))
        try:
            priorities = set([int(goal.priority) for goal in itertools.chain(goals, path_goals) if not goal.is_empty])
        except ValueError:
            raise Exception("GoalProgrammingMixin: All goal priorities must be of type int or castable to int")

        for priority in sorted(priorities):
            subproblems.append((priority, [goal for goal in goals if int(goal.priority) == priority and not goal.is_empty], [
                               goal for goal in path_goals if int(goal.priority) == priority and not goal.is_empty]))

        # Solve the subproblems one by one
        logger.info("Starting goal programming")

        success = False

        times = self.times()

        options = self.goal_programming_options()

        self.__subproblem_constraints = [OrderedDict() for ensemble_member in range(self.ensemble_size)]
        self.__subproblem_path_constraints = [OrderedDict() for ensemble_member in range(self.ensemble_size)]
        self.__first_run = True
        self.__results_are_current = False
        self.__original_constant_input_keys = {}
        for i, (priority, goals, path_goals) in enumerate(subproblems):
            logger.info("Solving goals at priority {}".format(priority))

            # Call the pre priority hook
            self.priority_started(priority)

            # Reset epsilons
            self.__subproblem_epsilons = []
            self.__subproblem_path_epsilons = []
            self.__subproblem_path_timeseries = []

            # Reset objective function
            self.__subproblem_objectives = []
            self.__subproblem_path_objectives = []

            for j, goal in enumerate(goals):
                if goal.critical:
                    if not goal.has_target_bounds:
                        raise Exception("Minimization goals cannot be critical")
                    epsilon = 0.0
                else:
                    if goal.has_target_bounds:
                        epsilon = ca.MX.sym('eps_{}_{}'.format(i, j))
                        self.__subproblem_epsilons.append(epsilon)

                if not goal.critical:
                    if goal.has_target_bounds:
                        self.__subproblem_objectives.append(lambda problem, ensemble_member, goal=goal, epsilon=epsilon: goal.weight * ca.constpow(
                            problem.extra_variable(epsilon.name(), ensemble_member=ensemble_member), goal.order))
                    else:
                        self.__subproblem_objectives.append(lambda problem, ensemble_member, goal=goal: goal.weight * ca.constpow(
                            goal.function(problem, ensemble_member) / (goal.function_range[1] - goal.function_range[0]) / goal.function_nominal, goal.order))

                if goal.has_target_bounds:
                    for ensemble_member in range(self.ensemble_size):
                        self.__add_goal_constraint(
                            goal, epsilon, ensemble_member, options)

            for j, goal in enumerate(path_goals):
                if goal.critical:
                    if not goal.has_target_bounds:
                        raise Exception("Minimization goals cannot be critical")
                    epsilon = np.zeros(len(self.times()))
                else:
                    if goal.has_target_bounds:
                        epsilon = ca.MX.sym('path_eps_{}_{}'.format(i, j))
                        self.__subproblem_path_epsilons.append(epsilon)

                if goal.has_target_min:
                    min_series = 'path_min_{}_{}'.format(i, j)

                    if isinstance(goal.target_min, Timeseries):
                        target_min = Timeseries(goal.target_min.times, goal.target_min.values)
                        target_min.values[np.logical_or(np.isnan(target_min.values), np.isneginf(target_min.values))] = -sys.float_info.max
                    else:
                        target_min = goal.target_min

                    self.__subproblem_path_timeseries.append(
                        (min_series, target_min))
                else:
                    min_series = None
                if goal.has_target_max:
                    max_series = 'path_max_{}_{}'.format(i, j)

                    if isinstance(goal.target_max, Timeseries):
                        target_max = Timeseries(goal.target_max.times, goal.target_max.values)
                        target_max.values[np.logical_or(np.isnan(target_max.values), np.isposinf(target_max.values))] = sys.float_info.max
                    else:
                        target_max = goal.target_max

                    self.__subproblem_path_timeseries.append(
                        (max_series, target_max))
                else:
                    max_series = None

                if not goal.critical:
                    if goal.has_target_bounds:
                        self.__subproblem_objectives.append(lambda problem, ensemble_member, goal=goal, epsilon=epsilon: goal.weight * ca.sum1(
                            ca.constpow(problem.state_vector(epsilon.name(), ensemble_member=ensemble_member), goal.order)))
                    else:
                        self.__subproblem_path_objectives.append(lambda problem, ensemble_member, goal=goal: goal.weight *
                            ca.constpow(goal.function(problem, ensemble_member) / (goal.function_range[1] - goal.function_range[0]) / goal.function_nominal, goal.order))

                if goal.has_target_bounds:
                    for ensemble_member in range(self.ensemble_size):
                        self.__add_path_goal_constraint(
                            goal, epsilon, ensemble_member, options, min_series, max_series)

            # Solve subproblem
            success = super().optimize(
                preprocessing=False, postprocessing=False, log_solver_failure_as_error=log_solver_failure_as_error)
            if not success:
                break

            self.__first_run = False

            # Store results.  Do this here, to make sure we have results even
            # if a subsequent priority fails.
            self.__results_are_current = False
            self.__results = [self.extract_results(
                ensemble_member) for ensemble_member in range(self.ensemble_size)]
            self.__results_are_current = True

            # Call the post priority hook, so that intermediate results can be
            # logged/inspected.
            self.priority_completed(priority)

            # Re-add constraints, this time with epsilon values fixed
            for ensemble_member in range(self.ensemble_size):
                for j, goal in enumerate(goals):
                    if goal.critical:
                        continue

                    if not goal.has_target_bounds or goal.violation_timeseries_id is not None or goal.function_value_timeseries_id is not None:
                        f = ca.Function('f', [self.solver_input], [goal.function(self, ensemble_member)])
                        function_value = float(f(self.solver_output))

                        # Store results
                        if goal.function_value_timeseries_id is not None:
                            self.set_timeseries(goal.function_value_timeseries_id, np.full_like(times, function_value), ensemble_member)

                    if goal.has_target_bounds:
                        epsilon = self.__results[ensemble_member][
                            'eps_{}_{}'.format(i, j)]

                        # Store results
                        # TODO tolerance
                        if goal.violation_timeseries_id is not None:
                            epsilon_active = epsilon
                            w = True
                            w &= (not goal.has_target_min or function_value / goal.function_nominal > goal.target_min / goal.function_nominal + options['interior_distance'])
                            w &= (not goal.has_target_max or function_value / goal.function_nominal < goal.target_max / goal.function_nominal - options['interior_distance'])
                            if w:
                                epsilon_active = np.nan
                            self.set_timeseries(goal.violation_timeseries_id, np.full_like(times, epsilon_active), ensemble_member)

                        # Add a relaxation to appease the barrier method.
                        epsilon += options['constraint_relaxation']
                    else:
                        epsilon = function_value

                    # Add inequality constraint
                    self.__add_goal_constraint(
                        goal, epsilon, ensemble_member, options)

                for j, goal in enumerate(path_goals):
                    if goal.critical:
                        continue

                    if not goal.has_target_bounds or goal.violation_timeseries_id is not None or goal.function_value_timeseries_id is not None:
                        # Compute path expression
                        expr = self.map_path_expression(goal.function(self, ensemble_member), ensemble_member)
                        f = ca.Function('f', [self.solver_input], [expr])
                        function_value = np.array(f(self.solver_output)).ravel()

                        # Store results
                        if goal.function_value_timeseries_id is not None:
                            self.set_timeseries(goal.function_value_timeseries_id, function_value, ensemble_member)

                    if goal.has_target_bounds:
                        epsilon = self.__results[ensemble_member][
                            'path_eps_{}_{}'.format(i, j)]

                        # Store results
                        if goal.violation_timeseries_id is not None:
                            epsilon_active = np.copy(epsilon)
                            m = goal.target_min
                            if isinstance(m, Timeseries):
                                m = self.interpolate(times, goal.target_min.times, goal.target_min.values)
                            M = goal.target_max
                            if isinstance(M, Timeseries):
                                M = self.interpolate(times, goal.target_max.times, goal.target_max.values)
                            w = np.ones_like(times)
                            if goal.has_target_min:
                                w = np.logical_and(w, np.logical_or(np.logical_not(np.isfinite(m)), function_value / goal.function_nominal > m / goal.function_nominal + options['interior_distance']))
                            if goal.has_target_max:
                                w = np.logical_and(w, np.logical_or(np.logical_not(np.isfinite(M)), function_value / goal.function_nominal < M / goal.function_nominal - options['interior_distance']))
                            epsilon_active[w] = np.nan
                            self.set_timeseries(goal.violation_timeseries_id, epsilon_active, ensemble_member)

                        # Add a relaxation to appease the barrier method.
                        epsilon += options['constraint_relaxation']
                    else:
                        epsilon = function_value

                    # Add inequality constraint
                    self.__add_path_goal_constraint(
                        goal, epsilon, ensemble_member, options)

        logger.info("Done goal programming")

        # Do post-processing
        if postprocessing:
            self.post()

        # Done
        return success

    def priority_started(self, priority: int) -> None:
        """
        Called when optimization for goals of certain priority is started.

        :param priority: The priority level that was started.
        """
        pass

    def priority_completed(self, priority: int) -> None:
        """
        Called after optimization for goals of certain priority is completed.

        :param priority: The priority level that was completed.
        """
        pass

    def extract_results(self, ensemble_member=0):
        if self.__results_are_current:
            logger.debug("Returning cached results")
            return self.__results[ensemble_member]

        # If self.__results is not up to date, do the super().extract_results
        # method
        return super().extract_results(ensemble_member)
