from rtctools.optimization.collocated_integrated_optimization_problem \
    import CollocatedIntegratedOptimizationProblem
from rtctools.optimization.goal_programming_mixin \
    import GoalProgrammingMixin, Goal
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.util import run_optimization_problem
from numpy import inf


class WaterLevelRangeGoal(Goal):
    # Applying a goal to every time step is easily done by returning the
    # optimization_problem.state('var') method and then passing the goal using
    # the path_goals() method. Note that each timestep is implemented as an
    # independant goal- if we cannot satisfy our min/max on time step A, it will
    # not affect our desire to satisfy the goal at time step B.
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state('storage.HQ.H')

    # Enclosure of the function range.
    function_range = (0, 0.5)

    # One goal can introduce a single or two constraints (min and/or max). Our
    # target water level range is 0.43 - 0.44. We might not always be able to
    # realize this, but we want to try.
    target_min = 0.43
    target_max = 0.44

    # Because we want to satisfy our water level target first, this has a
    # higher priority (=lower number).
    priority = 1


class MinimizeQpumpGoal(Goal):
    # If we do not specify any minimum or maximum value in this class, the
    # goal programming mixin will try to minimize the following function.
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state('Q_pump')

    # Every goal needs a rough (over)estimate (enclosure) of the range of the
    # function defined above.
    function_range = (0, 10.0)
    # The lower the number returned by this function, the higher the priority.
    priority = 2
    # The penalty variable is taken to the order'th power.
    order = 1


class MinimizeChangeInQpumpGoal(Goal):
    # To reduce pump power cycles, we add a third goal to minimize changes in
    # Q_pump.
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.der('Q_pump')
    function_range = (-100.0, 100.0)
    priority = 3
    order = 2


class Example(GoalProgrammingMixin, CSVMixin, ModelicaMixin,
              CollocatedIntegratedOptimizationProblem):
    """
    An introductory example to goal programming in RCT-Tools
    """
    def path_constraints(self, ensemble_member):
        # We want to add a few hard constraints to our problem. The goal
        # programming mixin however also generates constraints (and objectives)
        # from on our goals, so we have to call super() here.
        constraints = super(Example, self).path_constraints(ensemble_member)

        # Release through orifice downhill only. This constraint enforces the
        # fact that water only flows downhill
        constraints.append((self.state('Q_orifice') +
                           (1 - self.state('is_downhill')) * 10, 0.0, 10.0))

        # Make sure is_downhill is true only when the sea is lower than the
        # water level in the storage.
        M = 1e10  # M is a handy big number
        constraints.append((self.state('H_sea') - self.state('storage.HQ.H') -
                           (1 - self.state('is_downhill')) * M, -inf, 0.0))
        constraints.append((self.state('H_sea') - self.state('storage.HQ.H') +
                           self.state('is_downhill') * M, 0.0, inf))

        # Orifice flow constraint. Uses the equation:
        # Q(HUp, HDown, d) = width * C * d * (2 * g * (HUp - HDown)) ^ 0.5
        # Note that this equation is only valid for orifices that are submerged
                  # units:  description:
        w = 3.0   # m       width of orifice
        d = 0.8   # m       hight of orifice
        C = 1.0   # none    orifice constant
        g = 9.8   # m/s^2   gravitational acceleration
        constraints.append(
            (((self.state('Q_orifice') / (w * C * d)) ** 2) / (2 * g) +
             self.state('orifice.HQDown.H') - self.state('orifice.HQUp.H') -
             M * (1 - self.state('is_downhill')),
            -inf, 0.0))

        return constraints

    def path_goals(self):
        # Sorting goals on priority is done in the goal programming mixin. We
        # do not have to worry about order here.
        return [WaterLevelRangeGoal(), MinimizeQpumpGoal(), MinimizeChangeInQpumpGoal()]

    def priority_completed(self, priority):
        # We want to show that the results of our highest priority goal (water
        # level) are remembered. The other information we want to see is how our
        # lower priority goal (Q_pump) progresses. We can write some code that
        # sumerizes the results and stores it.

        # A little bit of tolerance when checking for acceptance, because
        # strictly speaking 0.4299... is smaller than 0.43.
        _min = 0.43 - 1e-4
        _max = 0.44 + 1e-4

        results = self.extract_results()
        n_level_satisfied = sum(
            [1 for x in results['storage.HQ.H'] if _min <= x <= _max])
        q_pump_integral = sum(results['Q_pump'])
        self.intermediate_results.append(
            (priority, n_level_satisfied, q_pump_integral))

    def pre(self):
        # Call super() class to not overwrite default behaviour
        super(Example, self).pre()
        # We keep track of our intermediate results, so that we can print some
        # information about the progress of goals at the end of our run.
        self.intermediate_results = []

    def post(self):
        # Call super() class to not overwrite default behaviour
        super(Example, self).post()
        for priority, n_level_satisfied, q_pump_integral in self.intermediate_results:
            print("\nAfter finishing goals of priority {}:".format(priority))
            print("Level goal satisfied at {} of {} time steps".format(
                n_level_satisfied, len(self.times())))
            print("Integral of Q_pump = {:.2f}".format(q_pump_integral))

    # Any solver options can be set here
    def solver_options(self):
        options = super(Example, self).solver_options()
        options['print_level'] = 1
        return options

# Run
run_optimization_problem(Example, base_folder='..')
