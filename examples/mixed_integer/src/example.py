from rtctools.optimization.collocated_integrated_optimization_problem \
    import CollocatedIntegratedOptimizationProblem
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.util import run_optimization_problem
from numpy import inf


class Example(CSVMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    """
    This class is the optimization problem for the Example. Within this class,
    the objective, constraints and other options are defined.
    """

    # This is a method that returns an expression for the objective function.
    # RTC-Tools always minimizes the objective.
    def objective(self, ensemble_member):
        # Minimize water pumped. The total water pumped is the integral of the
        # water pumped from the starting time until the stoping time. In
        # practice, self.integral() is a summation of all the discrete states.
        return self.integral('Q_pump', ensemble_member)

    # A path constraint is a constraint where the values in the constraint are a
    # Timeseries rather than a single number.
    def path_constraints(self, ensemble_member):
        # Call super to get default constraints
        constraints = super(Example, self).path_constraints(ensemble_member)
        # M is a handy big number
        M = 1e10

        # Release through orifice downhill only. This constraint enforces the
        # fact that water only flows downhill.
        constraints.append(
            (self.state('Q_orifice') + (1 - self.state('is_downhill')) * 10,
             0.0, 10.0))

        # Make sure is_downhill is true only when the sea is lower than the
        # water level in the storage.
        constraints.append((self.state('H_sea') - self.state('storage.HQ.H') -
                            (1 - self.state('is_downhill')) * M, -inf, 0.0))
        constraints.append((self.state('H_sea') - self.state('storage.HQ.H') +
                            self.state('is_downhill') * M, 0.0, inf))

        # Orifice flow upper bound. Uses the equation:
        # Q = width * C * height * (2 * g * (HQUp.H - HQDown.H)) ^ 0.5
        # orifice.LHS is the left-hand-side of this equation in standard form:
        # ((Q / (width * height * C)) ^ 2) / (g * 2) + HQDown.H - HQUp.H = 0
        constraints.append(
            (self.state('orifice.LHS') - M * (1.0 - self.state('is_downhill')),
             -inf, 0.0))
        # Note that this element is only valid for orifices that are submerged!

        return constraints

    # Any solver options can be set here
    def solver_options(self):
        options = super(Example, self).solver_options()
        # Restrict solver output
        options['print_level'] = 1
        return options

# Run
run_optimization_problem(Example)