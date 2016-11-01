# cython: embedsignature=True

from casadi import MX, MXFunction, sumRows, substitute, constpow
from abc import ABCMeta, abstractmethod
import numpy as np
cimport numpy as np
import itertools
import logging
import cython
import uuid
import sys
from sets import Set

from optimization_problem import OptimizationProblem
from timeseries import Timeseries

logger = logging.getLogger("rtctools")


class Goal(object):
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

    __metaclass__ = ABCMeta

    @abstractmethod
    def function(self, optimization_problem, ensemble_member):
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

    @property
    def has_target_min(self):
        """
        ``True`` if the user goal has min bounds.
        """
        if isinstance(self.target_min, Timeseries):
            return np.any(np.isfinite(self.target_min.values))
        else:
            return np.isfinite(self.target_min)

    @property
    def has_target_max(self):
        """
        ``True`` if the user goal has max bounds.
        """
        if isinstance(self.target_max, Timeseries):
            return np.any(np.isfinite(self.target_max.values))
        else:
            return np.isfinite(self.target_max)

    @property
    def has_target_bounds(self):
        """
        ``True`` if the user goal has min/max bounds.
        """
        return (self.has_target_min or self.has_target_max)

    def get_dependency_key(self):
        """
        Returns the "dependency key".  Goals with same dependency key are assumed to determine the same states.  This
        is used to eliminate linearly dependent constraints from the optimization problem.
        """
        if hasattr(self, 'dependency_key'):
            return self.dependency_key

        self.dependency_key = uuid.uuid4()

        return self.dependency_key


class GoalProgrammingMixin(OptimizationProblem):
    """
    Adds lexicographic goal programming to your optimization problem.
    """

    __metaclass__ = ABCMeta

    class _GoalConstraint:

        def __init__(self, goal, function, min, max):
            self.goal = goal
            self.function = function
            self.min = min
            self.max = max

    @property
    def extra_variables(self):
        return self._subproblem_epsilons

    @property
    def path_variables(self):
        return self._subproblem_path_epsilons + [variable for (variable, value) in self._subproblem_path_timeseries]

    def bounds(self):
        bounds = super(GoalProgrammingMixin, self).bounds()
        for epsilon in self._subproblem_epsilons + self._subproblem_path_epsilons:
            bounds[epsilon.getName()] = (0.0, 1.0)
        for (variable, value) in self._subproblem_path_timeseries:
            # IPOPT will turn these variables into parameters (assuming we
            # remain in 'make_parameter' mode)
            bounds[variable.getName()] = (value, value)
        return bounds

    def seed(self, ensemble_member):
        if self._first_run:
            seed = super(GoalProgrammingMixin, self).seed(ensemble_member)
        else:
            # Seed with previous results
            seed = {}
            epsilons = Set(map(lambda sym: sym.getName(
            ), self._subproblem_epsilons + self._subproblem_path_epsilons))
            for key in self._results[ensemble_member].keys():
                if key in epsilons:
                    continue  # Don't seed epsilon values.
                seed[key] = Timeseries(self.times(key), self._results[
                                       ensemble_member][key])

        # Seed epsilons
        for epsilon in self._subproblem_epsilons:
            seed[epsilon.getName()] = 1.0

        times = self.times()
        for epsilon in self._subproblem_path_epsilons:
            seed[epsilon.getName()] = Timeseries(times, np.ones(len(times)))

        return seed

    def objective(self, ensemble_member):
        acc_objective = MX(0)
        for o in self._subproblem_objectives:
            acc_objective += o(self, ensemble_member)
        return acc_objective / len(self._subproblem_objectives)

    def constraints(self, ensemble_member):
        constraints = []
        for l in self._subproblem_constraints[ensemble_member].values():
            constraints.extend(map(lambda constraint: (
                constraint.function(self), constraint.min, constraint.max), l))
        # Enforce a consistent ordering for the constraints, so that the ordering is not dependent on randomly
        # generated UUIDs.  See RTCTOOLS-485.
        constraints.sort(key=lambda constraint: ','.join(
            [repr(constraint[0]), str(constraint[1]), str(constraint[2])]))
        return constraints

    def path_constraints(self, ensemble_member):
        path_constraints = []
        for l in self._subproblem_path_constraints[ensemble_member].values():
            path_constraints.extend(map(lambda constraint: (
                constraint.function(self), constraint.min, constraint.max), l))
        # Enforce a consistent ordering for the constraints, so that the ordering is not dependent on randomly
        # generated UUIDs.  See RTCTOOLS-485.
        path_constraints.sort(key=lambda constraint: ','.join(
            [repr(constraint[0]), str(constraint[1]), str(constraint[2])]))
        return path_constraints

    def solver_options(self):
        # Call parent
        options = super(GoalProgrammingMixin, self).solver_options()

        # Make sure constant states, such as min/max timeseries for violation variables,
        # are turned into parameters for the final optimization problem.
        options['fixed_variable_treatment'] = 'make_parameter'

        if not self.goal_programming_options()['mu_reinit']:
            options['mu_strategy'] = 'monotone'
            options['gather_stats'] = True
            if not self._first_run:
                options['mu_init'] = self.nlp_solver.getStat('iterations')[
                    'mu'][-1]

        # Done
        return options

    def goal_programming_options(self):
        """
        Returns a dictionary of options controlling the goal programming process.

        +---------------------------+-----------+---------------+
        | Option                    | Type      | Default value |
        +===========================+===========+===============+
        | ``constraint_relaxation`` | ``float`` | ``0.0``       |
        +---------------------------+-----------+---------------+
        | ``violation_tolerance``   | ``float`` | ``1.0``       |
        +---------------------------+-----------+---------------+
        | ``mu_reinit``             | ``bool``  | ``True``      |
        +---------------------------+-----------+---------------+

        Constraints generated by the goal programming algorithm are relaxed by applying the specified relaxation.
        Use of this option is normally not required.

        A goal is considered to be violated if the violation, scaled between 0 and 1, is greater than the specified tolerance.
        Violated goals are fixed.  Use of this option is normally not required.

        The Ipopt barrier parameter ``mu`` is normally re-initialized a every iteration of the goal programming algorithm, unless
        mu_reinit is set to ``False``.  Use of this option is normally not required.

        :returns: A dictionary of goal programming options.
        """

        options = {}

        options['mu_reinit'] = True
        options['constraint_relaxation'] = 0.0  # Disable by default
        options['violation_tolerance'] = 1.0  # Disable by default

        return options

    def goals(self):
        """
        User problem returns list of :class:`Goal` objects.

        :returns: A list of goals.
        """
        return []

    def path_goals(self):
        """
        User problem returns list of path :class:`Goal` objects.

        :returns: A list of path goals.
        """
        return []

    def _add_goal_constraint(self, goal, epsilon, ensemble_member, options):
        # Check existing constraints for this state.
        constraints = self._subproblem_constraints[
            ensemble_member].get(goal.get_dependency_key(), [])
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
                        "Minimum value of goal less than minimum of a higher priority goal")
            elif constraint.goal.has_target_max:
                if goal.target_max > constraint.goal.target_max:
                    raise Exception(
                        "Maximum value of goal greater than maximum of a higher priority goal")

        # Check goal consistency
        if goal.has_target_min and goal.has_target_max:
            if goal.target_min > goal.target_max:
                raise Exception("Target minimum exceeds target maximum for goal {}".format(goal))

        if isinstance(epsilon, MX):
            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation
                # variables epsilon bounded between 0 and 1.
                if goal.has_target_min:
                    constraint = self._GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: (goal.function(problem, ensemble_member) - problem.extra_variable(
                        epsilon.getName(), ensemble_member=ensemble_member) * (goal.function_range[0] - goal.target_min) - goal.target_min) / goal.function_nominal, 0.0, np.inf)
                    constraints.append(constraint)
                if goal.has_target_max:
                    constraint = self._GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: (goal.function(problem, ensemble_member) - problem.extra_variable(
                        epsilon.getName(), ensemble_member=ensemble_member) * (goal.function_range[1] - goal.target_max) - goal.target_max) / goal.function_nominal, -np.inf, 0.0)
                    constraints.append(constraint)
            else:
                # Epsilon encodes the position within the function range,
                # scaled between 0 and 1.
                constraint = self._GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: (goal.function(problem, ensemble_member) - problem.extra_variable(
                    epsilon.getName(), ensemble_member=ensemble_member) * (goal.function_range[1] - goal.function_range[0]) - goal.function_range[0]) / goal.function_nominal, 0.0, 0.0)
                constraints.append(constraint)

            # TODO forgetting max like this.
            # Epsilon is not fixed yet.  This constraint is therefore linearly independent of any existing constraints,
            # and we add it to the list of constraints for this state.  We keep the existing constraints to ensure
            # that the attainment of previous goals is not worsened.
            self._subproblem_constraints[ensemble_member][
                goal.get_dependency_key()] = constraints
        else:
            fix_value = False

            constraint = self._GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal: goal.function(
                problem, ensemble_member) / goal.function_nominal, -np.inf, np.inf)
            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation
                # variables epsilon bounded between 0 and 1.
                if epsilon <= options['violation_tolerance']:
                    if goal.has_target_min:
                        constraint.min = (
                            epsilon * (goal.function_range[0] - goal.target_min) + goal.target_min) / goal.function_nominal
                    if goal.has_target_max:
                        constraint.max = (
                            epsilon * (goal.function_range[1] - goal.target_max) + goal.target_max) / goal.function_nominal
                else:
                    # Equality constraint to optimized value
                    fix_value = True

                    function = MXFunction('function', [self.solver_input], [
                                          goal.function(self, ensemble_member)])
                    [value] = function.call([self.solver_output])

                    constraint.min = value / goal.function_nominal
                    constraint.max = value / goal.function_nominal
            else:
                # Epsilon encodes the position within the function range,
                # scaled between 0 and 1.
                fix_value = True

                constraint.min = (epsilon * (goal.function_range[1] - goal.function_range[
                                  0]) + goal.function_range[0]) / goal.function_nominal
                constraint.max = (epsilon * (goal.function_range[1] - goal.function_range[
                                  0]) + goal.function_range[0]) / goal.function_nominal

            # Epsilon is fixed.  Override previous {min,max} constraints for
            # this state.
            new_constraints = [constraint]
            if not fix_value:
                for existing_constraint in constraints:
                    if goal.has_target_min and existing_constraint.goal.has_target_min:
                        # We have an existing min constraint, and are adding a new min constraint.
                        # Skip the existing constraint.
                        continue
                    if goal.has_target_max and existing_constraint.goal.has_target_max:
                        # We have an existing max constraint, and are adding a new max constraint.
                        # Skip the existing constraint.
                        continue
                    new_constraints.append(existing_constraint)
            self._subproblem_constraints[ensemble_member][
                goal.get_dependency_key()] = new_constraints

    def _add_path_goal_constraint(self, goal, epsilon, ensemble_member, options, min_series=None, max_series=None):
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
        constraints = self._subproblem_path_constraints[
            ensemble_member].get(goal.get_dependency_key(), [])
        for constraint in constraints:
            if constraint.goal == goal:
                continue
            constraint_m, constraint_M = _min_max_arrays(constraint.goal)
            if np.all(goal_m == goal_M):
                # This variable has already been fixed.  Don't add new
                # constraints for it.
                return
            elif constraint.goal.has_target_min:
                if np.any(goal_m < constraint_m):
                    raise Exception(
                        "Minimum value of goal less than minimum of a higher priority goal")
            elif constraint.goal.has_target_max:
                if np.any(goal_M > constraint_M):
                    raise Exception(
                        "Maximum value of goal greater than maximum of a higher priority goal")

        # Check goal consistency
        if goal.has_target_min and goal.has_target_max:
            indices = np.where(np.logical_and(
                np.isfinite(goal_m), np.isfinite(goal_M)))
            if np.any(goal_m[indices] > goal_M[indices]):
                raise Exception("Target minimum exceeds target maximum for goal {}".format(goal))

        if isinstance(epsilon, MX):
            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation
                # variables epsilon bounded between 0 and 1.
                if goal.has_target_min:
                    constraint = self._GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: (goal.function(problem, ensemble_member) - problem.variable(
                        epsilon.getName()) * (goal.function_range[0] - problem.variable(min_series.getName())) - problem.variable(min_series.getName())) / goal.function_nominal, 0.0, np.inf)
                    constraints.append(constraint)
                if goal.has_target_max:
                    constraint = self._GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: (goal.function(problem, ensemble_member) - problem.variable(
                        epsilon.getName()) * (goal.function_range[1] - problem.variable(max_series.getName())) - problem.variable(max_series.getName())) / goal.function_nominal, -np.inf, 0.0)
                    constraints.append(constraint)
            else:
                # Epsilon encodes the position within the function range,
                # scaled between 0 and 1.
                constraint = self._GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: (goal.function(
                    problem, ensemble_member) - problem.variable(epsilon.getName()) * (goal.function_range[1] - goal.function_range[0]) - goal.function_range[0]) / goal.function_nominal, 0.0, 0.0)
                constraints.append(constraint)

            # TODO forgetting max like this.
            # Epsilon is not fixed yet.  This constraint is therefore linearly independent of any existing constraints,
            # and we add it to the list of constraints for this state.  We keep the existing constraints to ensure
            # that the attainment of previous goals is not worsened.
            self._subproblem_path_constraints[ensemble_member][
                goal.get_dependency_key()] = constraints
        else:
            fix_value = False

            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation
                # variables epsilon bounded between 0 and 1.
                m, M = -np.inf * \
                    np.ones(len(times)), np.inf * np.ones(len(times))

                # Compute each min, max value separately for every time step
                for i, t in enumerate(times):
                    if epsilon[i] <= options['violation_tolerance']:
                        if np.isfinite(goal_m[i]):
                            m[i] = (epsilon[i] * (goal.function_range[0] -
                                                  goal_m[i]) + goal_m[i]) / goal.function_nominal
                        if np.isfinite(goal_M[i]):
                            M[i] = (epsilon[i] * (goal.function_range[1] -
                                                  goal_M[i]) + goal_M[i]) / goal.function_nominal
                    else:
                        # Equality constraint to optimized value
                        # TODO this does not perform well.
                        variables = self.dae_variables['states'] + self.dae_variables[
                            'algebraics'] + self.dae_variables['control_inputs'] + self.dae_variables['constant_inputs']
                        values = [self.state_at(
                            variable, t, ensemble_member=ensemble_member) for variable in variables]
                        [function] = substitute(
                            [goal.function(self, ensemble_member)], variables, values)
                        function = MXFunction(
                            'function', [self.solver_input], [function])
                        [value] = function.call([self.solver_output])

                        m[i] = value / goal.function_nominal
                        M[i] = value / goal.function_nominal
            else:
                # Epsilon encodes the position within the function range,
                # scaled between 0 and 1.
                fix_value = True

                m = (epsilon * (goal.function_range[1] - goal.function_range[
                     0]) + goal.function_range[0]) / goal.function_nominal
                M = (epsilon * (goal.function_range[1] - goal.function_range[
                     0]) + goal.function_range[0]) / goal.function_nominal

            constraint = self._GoalConstraint(goal, lambda problem, ensemble_member=ensemble_member, goal=goal: goal.function(
                problem, ensemble_member) / goal.function_nominal, Timeseries(times, m), Timeseries(times, M))

            # Epsilon is fixed.  Override previous {min,max} constraints for
            # this state.
            new_constraints = [constraint]
            if not fix_value:
                for existing_constraint in constraints:
                    existing_constraint_m, existing_constraint_M = _min_max_arrays(
                        existing_constraint.goal)
                    if np.all(np.isfinite(goal_m)) and np.all(np.isfinite(existing_constraint_m)):
                        # We have an existing min constraint, and are adding a new min constraint.
                        # Skip the existing constraint.
                        continue
                    if np.all(np.isfinite(goal_M)) and np.all(np.isfinite(existing_constraint_M)):
                        # We have an existing max constraint, and are adding a new max constraint.
                        # Skip the existing constraint.
                        continue
                    new_constraints.append(existing_constraint)
            self._subproblem_path_constraints[ensemble_member][
                goal.get_dependency_key()] = new_constraints

    def optimize(self, preprocessing=True, postprocessing=True):
        # Do pre-processing
        self.pre()

        # Group goals into subproblems
        subproblems = []
        goals = self.goals()
        path_goals = self.path_goals()

        # Validate goal definitions
        for goal in itertools.chain(goals, path_goals):
            if not np.isfinite(goal.function_range[0]) or not np.isfinite(goal.function_range[1]):
                raise Exception("No function range specified for goal {}".format(goal))

            if goal.target_min:
                if isinstance(goal.target_min, Timeseries):
                    if np.any(np.isfinite(goal.target_min.values)) and np.any(np.isnan(goal.target_min.values)):
                        raise Exception("target_min time series contains NaN for goal {}".format(goal))

            if goal.target_max:
                if isinstance(goal.target_max, Timeseries):
                    if np.any(np.isfinite(goal.target_max.values)) and np.any(np.isnan(goal.target_max.values)):
                        raise Exception("target_max time series contains NaN for goal {}".format(goal))

        priorities = Set([goal.priority for goal in itertools.chain(goals, path_goals)])
        for priority in sorted(priorities):
            subproblems.append((priority, [goal for goal in goals if goal.priority == priority], [
                               goal for goal in path_goals if goal.priority == priority]))

        # Solve the subproblems one by one
        logger.info("Starting goal programming")

        success = False

        options = self.goal_programming_options()

        self._subproblem_constraints = [{} for ensemble_member in range(self.ensemble_size)]
        self._subproblem_path_constraints = [{} for ensemble_member in range(self.ensemble_size)]
        self._first_run = True
        self._results_are_current = False
        for i, (priority, goals, path_goals) in enumerate(subproblems):
            logger.info("Solving goals at priority {}".format(priority))

            # Reset epsilons
            self._subproblem_epsilons = []
            self._subproblem_path_epsilons = []
            self._subproblem_path_timeseries = []

            # Reset objective function
            self._subproblem_objectives = []

            for j, goal in enumerate(goals):
                if goal.critical:
                    if not goal.has_target_bounds:
                        raise Exception("Minimization goals cannot be critical")
                    epsilon = 0.0
                else:
                    epsilon = MX.sym('eps_{}_{}'.format(i, j))
                    self._subproblem_epsilons.append(epsilon)

                self._subproblem_objectives.append(lambda problem, ensemble_member, goal=goal, epsilon=epsilon: goal.weight * constpow(
                    problem.extra_variable(epsilon.getName(), ensemble_member=ensemble_member), goal.order))

                for ensemble_member in range(self.ensemble_size):
                    self._add_goal_constraint(
                        goal, epsilon, ensemble_member, options)

            for j, goal in enumerate(path_goals):
                if goal.critical:
                    if not goal.has_target_bounds:
                        raise Exception("Minimization goals cannot be critical")
                    epsilon = 0.0
                else:
                    epsilon = MX.sym('path_eps_{}_{}'.format(i, j))
                    self._subproblem_path_epsilons.append(epsilon)

                if goal.has_target_min:
                    min_series = MX.sym('path_min_{}_{}'.format(i, j))
                    self._subproblem_path_timeseries.append(
                        (min_series, goal.target_min))
                else:
                    min_series = None
                if goal.has_target_max:
                    max_series = MX.sym('path_max_{}_{}'.format(i, j))
                    self._subproblem_path_timeseries.append(
                        (max_series, goal.target_max))
                else:
                    max_series = None

                self._subproblem_objectives.append(lambda problem, ensemble_member, goal=goal, epsilon=epsilon: goal.weight * sumRows(
                    constpow(problem.state_vector(epsilon.getName(), ensemble_member=ensemble_member), goal.order)))

                for ensemble_member in range(self.ensemble_size):
                    self._add_path_goal_constraint(
                        goal, epsilon, ensemble_member, options, min_series, max_series)

            # Solve subproblem
            success = super(GoalProgrammingMixin, self).optimize(
                preprocessing=False, postprocessing=False)
            if not success:
                break

            self._first_run = False

            # Store results.  Do this here, to make sure we have results even
            # if a subsequent priority fails.
            self._results_are_current = False
            self._results = [self.extract_results(
                ensemble_member) for ensemble_member in range(self.ensemble_size)]
            self._results_are_current = True

            # Call the post priority hook, so that intermediate results can be
            # logged/inspected.
            self.priority_completed(priority)

            # Re-add constraints, this time with epsilon values fixed
            for ensemble_member in range(self.ensemble_size):
                for j, goal in enumerate(goals):
                    if goal.critical:
                        continue

                    epsilon = self._results[ensemble_member][
                        'eps_{}_{}'.format(i, j)]
                    if goal.has_target_bounds:
                        # Add a relaxation to appease the barrier method.
                        epsilon += options['constraint_relaxation']

                    # Add inequality constraint
                    self._add_goal_constraint(
                        goal, epsilon, ensemble_member, options)

                for j, goal in enumerate(path_goals):
                    if goal.critical:
                        continue

                    epsilon = self._results[ensemble_member][
                        'path_eps_{}_{}'.format(i, j)]
                    if goal.has_target_bounds:
                        # Add a relaxation to appease the barrier method.
                        epsilon += options['constraint_relaxation']

                    # Add inequality constraint
                    self._add_path_goal_constraint(
                        goal, epsilon, ensemble_member, options)

        logger.info("Done goal programming")

        # Do post-processing
        self.post()

        # Done
        return success

    def priority_completed(self, priority):
        """
        Called after optimization for goals of certain priority is completed.

        :param priority: The priority level that was completed.
        """
        pass

    def extract_results(self, ensemble_member=0):
        if self._results_are_current:
            logger.debug("Returning cached results")
            return self._results[ensemble_member]

        # If self._results is not up to date, do the super().extract_results
        # method
        return super(GoalProgrammingMixin, self).extract_results(ensemble_member)
