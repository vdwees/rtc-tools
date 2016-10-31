from rtctools.optimization.collocated_integrated_optimization_problem \
    import CollocatedIntegratedOptimizationProblem
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.util import run_optimization_problem
from numpy import inf, log, exp


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
        # # M is a handy big number
        # M = 1e10
        #
        # # Release through orifice downhill only. This constraint enforces the
        # # fact that water only flows downhill.
        # constraints.append(
        #     (self.state('Q_orifice') + (1 - self.state('is_downhill')) * 10,
        #      0.0, 10.0))
        #
        # # Make sure is_downhill is true only when the sea is lower than the
        # # water level in the storage.
        # constraints.append((self.state('H_sea') - self.state('storage.HQ.H') -
        #                     (1 - self.state('is_downhill')) * M, -inf, 0.0))
        # constraints.append((self.state('H_sea') - self.state('storage.HQ.H') +
        #                     self.state('is_downhill') * M, 0.0, inf))

        # Orifice flow constraint. Uses the equation:
        # Q(HUp, HDown, d) = width * C * d * (2 * g * (HUp - HDown)) ^ 0.5
        # Note that this equation is only valid for orifices that are submerged
                  # units:  description:
        w = 3.0   # m       width of orifice
        d = 0.8   # m       hight of orifice
        C = 1.0   # none    orifice constant
        g = 9.8   # m/s^2   gravitational acceleration
        # constraints.append(
        #     (((self.state('Q_orifice') / (w * C * d)) ** 2) / (2 * g) +
        #      self.state('orifice.HQDown.H') - self.state('orifice.HQUp.H') -
        #      M * (1 - self.state('is_downhill')),
        #     -inf, 0.0))

        factor = 100

        constraints.append(
            (((self.state('Q_orifice') / (w * C * d)) ** 2) / (2 * g) - log(exp(0) + exp(factor * (self.state('orifice.HQUp.H') - self.state('orifice.HQDown.H')))) / factor,
             -inf, 0.0))

        return constraints

    # Any solver options can be set here
    def solver_options(self):
        options = super(Example, self).solver_options()
        # Restrict solver output
        options['print_level'] = 1
        return options

# Run
run_optimization_problem(Example, base_folder='..')
