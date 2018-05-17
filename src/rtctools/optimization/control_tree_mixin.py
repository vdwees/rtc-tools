import logging
from typing import Dict, List, Union

import numpy as np

from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


class ControlTreeMixin(OptimizationProblem):
    """
    Adds a stochastic control tree to your optimization problem.
    """

    def control_tree_options(self) -> Dict[str, Union[List[str], List[float], int]]:
        """
        Returns a dictionary of options controlling the creation of a k-ary stochastic tree.

        +------------------------+---------------------+-----------------------+
        | Option                 | Type                | Default value         |
        +========================+=====================+=======================+
        | ``forecast_variables`` | ``list`` of strings | All constant inputs   |
        +------------------------+---------------------+-----------------------+
        | ``branching_times``    | ``list`` of floats  | ``self.times()``      |
        +------------------------+---------------------+-----------------------+
        | ``k``                  | ``int``             | ``2``                 |
        +------------------------+---------------------+-----------------------+

        A ``k``-ary tree is generated, branching at every interior branching time.
        Ensemble members are clustered to paths through the tree based on average
        distance over all forecast variables.

        :returns: A dictionary of control tree generation options.
        """

        options = {}

        options['forecast_variables'] = [var.name()
                                         for var in self.dae_variables['constant_inputs']]
        options['branching_times'] = self.times()[1:]
        options['k'] = 2

        return options

    def discretize_controls(self, resolved_bounds):
        # Collect options
        options = self.control_tree_options()

        # Make sure branching times contain initial and final time.  The
        # presence of these is assumed below.
        times = self.times()
        t0 = self.initial_time
        branching_times = options['branching_times']
        n_branching_times = len(branching_times)
        if n_branching_times > len(times) - 1:
            raise Exception("Too many branching points specified")
        branching_times = np.concatenate(([t0], branching_times, [np.inf]))

        logger.debug("ControlTreeMixin: Branching times:")
        logger.debug(branching_times)

        # Branches start at branching times, so that the tree looks like the following:
        #
        #         *-----
        #   *-----
        #         *-----
        #
        #   t0    t1
        #
        # with branching time t1.
        branches = {}

        def branch(current_branch):
            if len(current_branch) >= n_branching_times:
                return

            # Branch stats
            n_branch_members = len(branches[current_branch])
            if n_branch_members == 0:
                # Nothing to do
                return
            distances = np.zeros((n_branch_members, n_branch_members))

            # Decide branching on a segment of the time horizon
            branching_time_0 = branching_times[len(current_branch) + 1]
            branching_time_1 = branching_times[len(current_branch) + 2]

            # Compute reverse ensemble member index-to-distance index map.
            reverse = {}
            for i, member_i in enumerate(branches[current_branch]):
                reverse[member_i] = i

            # Compute distances between ensemble members, summed for all
            # forecast variables
            for forecast_variable in options['forecast_variables']:
                # We assume the time stamps of the forecasts in all ensemble
                # members to be identical
                timeseries = self.constant_inputs(ensemble_member=0)[
                    forecast_variable]
                els = np.logical_and(
                    timeseries.times >= branching_time_0, timeseries.times < branching_time_1)

                # Compute distance between ensemble members
                for i, member_i in enumerate(branches[current_branch]):
                    timeseries_i = self.constant_inputs(ensemble_member=member_i)[
                        forecast_variable]
                    for j, member_j in enumerate(branches[current_branch]):
                        timeseries_j = self.constant_inputs(ensemble_member=member_j)[
                            forecast_variable]
                        distances[
                            i, j] += np.linalg.norm(timeseries_i.values[els] - timeseries_j.values[els])

            # Keep track of ensemble members that have not yet been allocated
            # to a new branch
            available = set(branches[current_branch])

            idx = 0
            for i in range(options['k']):
                if idx >= 0:
                    branches[current_branch +
                             str(i)] = [branches[current_branch][idx]]

                    available.remove(branches[current_branch][idx])

                    # We select the scenario with the max min distance to the other branches
                    min_distances = np.array([
                        min([np.inf] + [distances[j, k]
                            for j, member_j in enumerate(branches[current_branch])
                            if member_j not in available and member_k in available])
                        for k, member_k in enumerate(branches[current_branch])
                        ], dtype=np.float64)
                    min_distances[np.where(min_distances == np.inf)] = -np.inf

                    idx = np.argmax(min_distances)
                    if min_distances[idx] <= 0:
                        idx = -1
                else:
                    branches[current_branch + str(i)] = []

            # Cluster remaining ensemble members to branches
            for member_i in available:
                min_i = 0
                min_distance = np.inf
                for i in range(options['k']):
                    branch2 = branches[current_branch + str(i)]
                    if len(branch2) > 0:
                        distance = distances[
                            reverse[member_i], reverse[branch2[0]]]
                        if distance < min_distance:
                            min_distance = distance
                            min_i = i
                branches[current_branch + str(min_i)].append(member_i)

            # Recurse
            for i in range(options['k']):
                branch(current_branch + str(i))

        current_branch = ''
        branches[current_branch] = list(range(self.ensemble_size))
        branch(current_branch)

        logger.debug("ControlTreeMixin:  Control tree is:")
        logger.debug(branches)

        # Map ensemble members to control inputs
        # (variable, (ensemble member, step)) -> control_index
        self.__control_indices = [{} for ensemble_member in range(self.ensemble_size)]
        count = 0
        for control_input in self.controls:
            times = self.times(control_input)
            for member in range(self.ensemble_size):
                self.__control_indices[member][control_input] = np.zeros(
                    len(times), dtype=np.int16)
            for branch, members in branches.items():
                branching_time_0 = branching_times[len(branch) + 0]
                branching_time_1 = branching_times[len(branch) + 1]
                els = np.logical_and(
                    times >= branching_time_0, times < branching_time_1)
                nnz = np.count_nonzero(els)
                for member in members:
                    self.__control_indices[member][control_input][els] = \
                        list(range(count, count + nnz))
                count += nnz

        # Construct bounds and initial guess
        discrete = np.zeros(count, dtype=np.bool)

        lbx = np.full(count, -np.inf, dtype=np.float64)
        ubx = np.full(count, np.inf, dtype=np.float64)

        x0 = np.zeros(count, dtype=np.float64)

        for ensemble_member in range(self.ensemble_size):
            seed = self.seed(ensemble_member)

            for variable in self.controls:
                times = self.times(variable)

                discrete[self.__control_indices[ensemble_member][variable]] = \
                    self.variable_is_discrete(variable)

                try:
                    bound = resolved_bounds[variable]
                except KeyError:
                    pass
                else:
                    nominal = self.variable_nominal(variable)
                    if bound[0] is not None:
                        if isinstance(bound[0], Timeseries):
                            lbx[self.__control_indices[ensemble_member][variable]] = self.interpolate(
                                times, bound[0].times, bound[0].values, -np.inf, -np.inf) / nominal
                        else:
                            lbx[self.__control_indices[ensemble_member][variable]] = bound[0] / nominal
                    if bound[1] is not None:
                        if isinstance(bound[1], Timeseries):
                            ubx[self.__control_indices[ensemble_member][variable]] = self.interpolate(
                                times, bound[1].times, bound[1].values, +np.inf, +np.inf) / nominal
                        else:
                            ubx[self.__control_indices[ensemble_member][variable]] = bound[1] / nominal

                    try:
                        seed_k = seed[variable]
                        x0[self.__control_indices[ensemble_member][variable]] = self.interpolate(
                            times, seed_k.times, seed_k.values, 0, 0) / nominal
                    except KeyError:
                        pass

        # Return number of control variables
        return count, discrete, lbx, ubx, x0, self.__control_indices

    def extract_controls(self, ensemble_member=0):
        # Solver output
        X = self.solver_output

        # Extract control inputs
        results = {}
        for variable in self.controls:
            results[variable] = np.array(self.variable_nominal(
                variable) * X[self.__control_indices[ensemble_member][variable], 0]).ravel()

        # Done
        return results

    def control_at(self, variable, t, ensemble_member=0, scaled=False, extrapolate=True):
        t0 = self.initial_time
        X = self.solver_input

        canonical, sign = self.alias_relation.canonical_signed(variable)
        for control_input in self.controls:
            times = self.times(control_input)
            if control_input == canonical:
                nominal = self.variable_nominal(control_input)
                variable_values = [X[i] for i in self.__control_indices[ensemble_member][
                    control_input]]
                f_left, f_right = np.nan, np.nan
                if t < t0:
                    history = self.history(ensemble_member)
                    try:
                        history_timeseries = history[control_input]
                    except KeyError:
                        if extrapolate:
                            sym = variable_values[0]
                        else:
                            sym = np.nan
                    else:
                        if extrapolate:
                            f_left = history_timeseries.values[0]
                            f_right = history_timeseries.values[-1]
                        sym = self.interpolate(
                                t, history_timeseries.times, history_timeseries.values, f_left, f_right)
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
                if sign < 0:
                    sym *= -1
                return sym

        raise KeyError(variable)
