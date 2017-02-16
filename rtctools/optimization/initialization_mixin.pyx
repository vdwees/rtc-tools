# cython: embedsignature=True

from rtctools.optimization.goal_programming_mixin import GoalProgrammingMixin, Goal
import logging

logger = logging.getLogger("rtctools")


class _MeasurementGoal(Goal):
    def __init__(self, state, measurement_id, max_deviation=1.0):
        self._state = state
        self._measurement_id = measurement_id

        self.function_range = (-max_deviation, max_deviation)
        self.function_nominal = max_deviation

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self._state, optimization_problem.initial_time, ensemble_member) - \
            optimization_problem.timeseries_at(self._measurement_id, optimization_problem.initial_time, ensemble_member)

    order = 2
    priority = -2


class _SmoothingGoal(Goal):
    def __init__(self, state1, state2, max_deviation=1.0):
        self._state1 = state1
        self._state2 = state2

        self.function_range = (-max_deviation, max_deviation)
        self.function_nominal = max_deviation

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self._state1, optimization_problem.initial_time, ensemble_member) - \
            optimization_problem.state(self._state2, optimization_problem.initial_time, ensemble_member)

    order = 2
    priority = -1


class InitializationMixin(GoalProgrammingMixin):
    def initial_state_measurements(self):
        """
        List of pairs (state, measurement_id) or triples (state, measurement_id, max_deviation).
        Default max_deviation is 1.
        """
        return []

    def initial_state_smoothing_pairs(self):
        """
        List of pairs (state1, state2) or triples (state1, state2, max_deviation).
        Default max_deviation is 1.
        """
        return []

    def goals(self, ensemble_member):
        g = super(InitializationMixin, self).goals(ensemble_member)

        for measurement in self.initial_state_measurements():
            g.append(_MeasurementGoal(*measurement))

        for smoothing_pair in self.initial_state_smoothing_pairs():
            g.append(_SmoothingGoal(*smoothing_pair))

        return g
