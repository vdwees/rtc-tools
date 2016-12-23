# cython: embedsignature=True

from optimization_problem import OptimizationProblem
from timeseries import Timeseries
from casadi import vertcat, MX
import numpy as np
import logging
import sys

logger = logging.getLogger("rtctools")


class ControlTreeMixin(OptimizationProblem):
    """
    Adds a stochastic control tree to your optimization problem.
    """

    def control_tree_options(self):
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

        options['forecast_variables'] = [var.getName()
                                         for var in self.dae_variables['constant_inputs']]
        options['branching_times'] = self.times()
        options['k'] = 2

        return options

    def discretize_controls(self):
        # Collect options
        options = self.control_tree_options()

        # Make sure braching times contain initial and final time.  The
        # presence of these is assumed below.
        t0 = self.initial_time
        tf = self.times()[-1]
        branching_times = options['branching_times']
        if branching_times[0] != t0:
            branching_times = np.concatenate(([t0], branching_times))
        if branching_times[-1] != tf:
            branching_times = np.concatenate((branching_times, [tf]))

        logger.debug("ControlTreeMixin: Branching times:")
        logger.debug(branching_times)

        branches = {}

        def branch(current_branch):
            # We branch at most len(branching_points) - 2 times, due to the
            # inclusion of t0 and tf.
            if len(current_branch) >= len(branching_times) - 2:
                return

            # Branch stats
            n_branch_members = len(branches[current_branch])
            if n_branch_members == 0:
                # Nothing to do
                return
            distances = np.zeros((n_branch_members, n_branch_members))

            branching_time_0 = branching_times[len(current_branch) + 0]
            branching_time_1 = branching_times[len(current_branch) + 1]

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
                    timeseries_i = self.constant_inputs(ensemble_member=i)[
                        forecast_variable]
                    for j, member_j in enumerate(branches[current_branch]):
                        timeseries_j = self.constant_inputs(ensemble_member=j)[
                            forecast_variable]
                        distances[
                            i, j] += np.linalg.norm(timeseries_i.values[els] - timeseries_j.values[els])

            # Keep track of ensemble members that have not yet been allocated
            # to a new branch
            available = set(branches[current_branch])

            sum_distances = [sum(distances[i, j] for j in range(
                n_branch_members) if j != i) for i in range(n_branch_members)]
            for i in range(options['k']):
                idx = np.argmax(sum_distances)
                if np.isfinite(sum_distances[idx]):
                    branches[current_branch +
                             str(i)] = [branches[current_branch][idx]]

                    available.remove(branches[current_branch][idx])

                    sum_distances[idx] = -np.inf
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
        branches[current_branch] = range(self.ensemble_size)
        branch(current_branch)

        logger.debug("ControlTreeMixin:  Control tree is:")
        logger.debug(branches)

        # Map ensemble members to control inputs
        # (variable, (ensemble member, step)) -> control_index
        self._control_indices = {}
        count = 0
        for control_input in self.controls:
            times = self.times(control_input)
            self._control_indices[control_input] = np.zeros(
                (self.ensemble_size, len(times)), dtype=np.int16)
            for branch, members in branches.iteritems():
                branching_time_0 = branching_times[len(branch) + 0]
                branching_time_1 = branching_times[len(branch) + 1]
                if len(branch) < len(branching_times) - 2:
                    els = np.logical_and(
                        times >= branching_time_0, times < branching_time_1)
                else:
                    # Make sure the final element is included.
                    els = times >= branching_time_0
                nnz = np.count_nonzero(els)
                for member in members:
                    self._control_indices[control_input][
                        member, els] = range(count, count + nnz)
                count += nnz

        # Construct bounds and initial guess
        bounds = self.bounds()

        discrete = np.zeros(count, dtype=np.bool)

        lbx = -sys.float_info.max * np.ones(count)
        ubx = sys.float_info.max * np.ones(count)

        x0 = np.zeros(count)

        for ensemble_member in range(self.ensemble_size):
            seed = self.seed(ensemble_member)

            for variable in self.controls:
                times = self.times(variable)

                discrete[self._control_indices[variable][
                    ensemble_member, :]] = self.variable_is_discrete(variable)

                for alias in self.variable_aliases(variable):
                    try:
                        bound = bounds[alias.name]
                    except KeyError:
                        continue

                    nominal = self.variable_nominal(variable)
                    if bound[0] != None:
                        if isinstance(bound[0], Timeseries):
                            lbx[self._control_indices[variable][ensemble_member, :]] = self.interpolate(
                                times, bound[0].times, bound[0].values, -sys.float_info.max, -sys.float_info.max) / nominal
                        else:
                            lbx[self._control_indices[variable][
                                ensemble_member, :]] = bound[0] / nominal
                    if bound[1] != None:
                        if isinstance(bound[1], Timeseries):
                            ubx[self._control_indices[variable][ensemble_member, :]] = self.interpolate(
                                times, bound[1].times, bound[1].values, +sys.float_info.max, +sys.float_info.max) / nominal
                        else:
                            ubx[self._control_indices[variable][
                                ensemble_member, :]] = bound[1] / nominal

                    try:
                        seed_k = seed[variable]
                        x0[self._control_indices[variable][ensemble_member, :]] = self.interpolate(
                            times, seed_k.times, seed_k.values, 0, 0) / nominal
                    except KeyError:
                        pass

                    break

        # Return number of control variables
        return count, discrete, lbx, ubx, x0

    def extract_controls(self, ensemble_member=0):
        # Solver output
        X = self.solver_output

        # Extract control inputs
        results = {}
        for variable in self.controls:
            results[variable] = np.array(self.variable_nominal(
                variable) * X[self._control_indices[variable][ensemble_member, :], 0]).ravel()

            for alias in self.variable_aliases(variable):
                if alias.name != variable:
                    results[alias.name] = alias.sign * results[variable]

        # Done
        return results

    def control_vector(self, variable, ensemble_member=0):
        X = self.solver_input
        
        if ensemble_member is None:
            if not hasattr(self,"_mycache"):
                self._mycache = {}
            name = "control_vector_%s" % variable
            if name in self._mycache:
                return self._mycache[name]
            else:
                r = MX.sym(name, *self._control_indices[variable][0, :].shape)
                self._mycache[name] = r
                return r
        else:
            return X[self._control_indices[variable][ensemble_member, :]]

    def control_at(self, variable, t, ensemble_member=0, scaled=False, extrapolate=True):
        t0 = self.initial_time
        X = self.solver_input

        for control_input in self.controls:
            times = self.times(control_input)
            for alias in self.variable_aliases(control_input):
                if alias.name == variable:
                    nominal = self.variable_nominal(control_input)
                    variable_values = [X[i] for i in self._control_indices[
                        control_input][ensemble_member, :]]
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

        raise KeyError(variable)
