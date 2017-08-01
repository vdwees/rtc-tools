from .data_path import data_path
from test_case import TestCase

from rtctools.optimization.collocated_integrated_optimization_problem import CollocatedIntegratedOptimizationProblem
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries
from casadi import MX, vertcat
from unittest import expectedFailure
import numpy as np
import logging
import time
import sys
import os

logger = logging.getLogger("rtctools")
logger.setLevel(logging.DEBUG)



class TestProblem(ModelicaMixin, CollocatedIntegratedOptimizationProblem):

    def __init__(self):
        super(TestProblem, self).__init__(input_folder=data_path(), output_folder=data_path(
        ), model_name='TestModelWithInitial', model_folder=data_path())

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        parameters['u_max'] = 2.0
        return parameters

    def pre(self):
        # Do nothing
        pass

    def constant_inputs(self, ensemble_member):
        # Constant inputs
        return {'constant_input': Timeseries(self.times(), 1 - self.times())}

    def seed(self, ensemble_member):
        # No particular seeding
        return {}

    def objective(self, ensemble_member):
        # Quadratic penalty on state 'x' at final time
        xf = self.state_at('x', self.times(
            'x')[-1], ensemble_member=ensemble_member)
        return xf**2

    def constraints(self, ensemble_member):
        # No additional constraints
        return []

    def post(self):
        # Do
        pass

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options['cache'] = False
        return compiler_options


class TestProblemNonConvex(TestProblem):

    def __init__(self, u_seed):
        super(TestProblemNonConvex, self).__init__()

        self.u_seed = u_seed

    def objective(self, ensemble_member):
        # Make two local optima, at xf=1.0 and at xf=-1.0.
        xf = self.state_at(
            'x', self.times()[-1], ensemble_member=ensemble_member)
        return (xf**2 - 1.0)**2

    @property
    def initial_residual(self):
        # Set the initial state for 'x' to the neutral point.
        residual = []
        for state in self.dae_variables['states']:
            residual.append(state)
        return vertcat(*residual)

    def seed(self, ensemble_member):
        # Seed the controls.
        return {'u': Timeseries(self.times(), self.u_seed)}


class TestProblemConstrained(TestProblem):

    def constraints(self, ensemble_member):
        # Constrain x(t=1.9)^2 >= 0.1.
        x = self.state_at(
            'x', self.times()[-1] - 0.1, ensemble_member=ensemble_member)
        f = x**2
        return [(f, 0.1, sys.float_info.max)]


class TestProblemTrapezoidal(TestProblem):

    @property
    def theta(self):
        return 0.5


class TestProblemShort(TestProblem):

    def times(self, variable=None):
        return np.linspace(0.0, 1.0, 2)


class TestProblemAggregation(TestProblem):

    def times(self, variable=None):
        if variable == 'u':
            return np.linspace(0.0, 1.0, 11)
        else:
            return np.linspace(0.0, 1.0, 21)


class TestProblemEnsemble(TestProblem):

    @property
    def ensemble_size(self):
        return 2

    def constant_inputs(self, ensemble_member):
        # Constant inputs
        if ensemble_member == 0:
            return {'constant_input': Timeseries(self.times(), np.linspace(1.0, 0.0, 21))}
        else:
            return {'constant_input': Timeseries(self.times(), np.linspace(1.0, 0.5, 21))}


class TestProblemAlgebraic(ModelicaMixin, CollocatedIntegratedOptimizationProblem):

    def __init__(self):
        super(TestProblemAlgebraic, self).__init__(input_folder=data_path(
        ), output_folder=data_path(), model_name='TestModelAlgebraic', model_folder=data_path())

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    def pre(self):
        # Do nothing
        pass

    def bounds(self):
        # Variable bounds
        return {'u': (-2.0, 2.0)}

    def seed(self, ensemble_member):
        # No particular seeding
        return {}

    def objective(self, ensemble_member):
        return self.integral('u')

    def constraints(self, ensemble_member):
        # No additional constraints
        return []

    def post(self):
        # Do
        pass

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options['cache'] = False
        return compiler_options


class TestProblemMixedInteger(ModelicaMixin, CollocatedIntegratedOptimizationProblem):

    def __init__(self):
        super(TestProblemMixedInteger, self).__init__(input_folder=data_path(
        ), output_folder=data_path(), model_name='TestModelMixedInteger', model_folder=data_path())

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    def pre(self):
        # Do nothing
        pass

    def seed(self, ensemble_member):
        # No particular seeding
        return {}

    def objective(self, ensemble_member):
        return self.integral('y')

    def constraints(self, ensemble_member):
        # No additional constraints
        return []

    def post(self):
        # Do
        pass

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options['cache'] = False
        return compiler_options


class TestModelicaMixin(TestCase):

    def setUp(self):
        self.problem = TestProblem()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(
            abs(self.problem.objective_value), 0.0, objective_value_tol)

    def test_ifelse(self):
        print(self.results['switched'])
        print(self.results['x'])
        self.assertEqual(self.results['switched'][0], 1.0)
        self.assertEqual(self.results['switched'][-1], 2.0)

    def test_output(self):
        self.assertAlmostEqual(self.results[
                               'x']**2 + np.sin(self.problem.times()), self.results['z'], self.tolerance)

    def test_algebraic(self):
        self.assertAlmostEqual(self.results[
                               'y'] + self.results['x'], np.ones(len(self.problem.times())) * 3.0, self.tolerance)

    def test_bounds(self):
        self.assertAlmostGreaterThan(self.results['u'], -2, self.tolerance)
        self.assertAlmostLessThan(self.results['u'], 2, self.tolerance)

    def test_constant_input(self):
        verify = np.linspace(1.0, 0.0, 21)
        self.assertAlmostEqual(
            self.results['constant_output'], verify, self.tolerance)

    def test_delayed_feedback(self):
        self.assertAlmostEqual(self.results['x_delayed'][
                               2:], self.results['x'][:-2], self.tolerance)

    def test_multiple_states(self):
        self.assertAlmostEqual(self.results['w'][0], 0.0, self.tolerance)
        self.assertAlmostEqual(self.results['w'][-1], 0.5917, 1e-4)

    @expectedFailure
    def test_states_in(self):
        states = list(self.problem.states_in('x', 0.05, 0.95))
        verify = []
        for t in self.problem.times()[1:-1]:
            verify.append(self.problem.state_at('x', t))
        self.assertEqual(repr(states), repr(verify))

        states = list(self.problem.states_in('x', 0.051, 0.951))
        verify = [self.problem.state_at('x', 0.051)]
        for t in self.problem.times()[2:-1]:
            verify.append(self.problem.state_at('x', t))
        verify.append(self.problem.state_at('x', 0.951))
        self.assertEqual(repr(states), repr(verify))

        states = list(self.problem.states_in('x', 0.0, 0.951))
        verify = []
        for t in self.problem.times()[0:-1]:
            verify.append(self.problem.state_at('x', t))
        verify.append(self.problem.state_at('x', 0.951))
        self.assertEqual(repr(states), repr(verify))

    def test_der(self):
        der = self.problem.der_at('x', 0.05)
        verify = (self.problem.state_at('x', 0.05) -
                  self.problem.state_at('x', 0.0)) / 0.05
        self.assertEqual(repr(der), repr(verify))

        der = self.problem.der_at('x', 0.051)
        verify = (self.problem.state_at('x', 0.1) -
                  self.problem.state_at('x', 0.05)) / 0.05
        self.assertEqual(repr(der), repr(verify))

    # This test fails, because we use CasADi sumRows() now.
    @expectedFailure
    def test_integral(self):
        integral = self.problem.integral('x', 0.05, 0.95)
        knots = self.problem.times()[1:-1]
        verify = MX(0.0)
        for i in xrange(len(knots) - 1):
            verify += 0.5 * (self.problem.state_at('x', knots[i]) + self.problem.state_at(
                'x', knots[i + 1])) * (knots[i + 1] - knots[i])
        self.assertEqual(repr(integral), repr(verify))

        integral = self.problem.integral('x', 0.051, 0.951)
        knots = []
        knots.append(0.051)
        knots.extend(self.problem.times()[2:-1])
        knots.append(0.951)
        verify = MX(0.0)
        for i in xrange(len(knots) - 1):
            verify += 0.5 * (self.problem.state_at('x', knots[i]) + self.problem.state_at(
                'x', knots[i + 1])) * (knots[i + 1] - knots[i])
        self.assertEqual(repr(integral), repr(verify))

        integral = self.problem.integral('x', 0.0, 0.951)
        knots = list(self.problem.times()[0:-1]) + [0.951]
        verify = MX(0.0)
        for i in xrange(len(knots) - 1):
            verify += 0.5 * (self.problem.state_at('x', knots[i]) + self.problem.state_at(
                'x', knots[i + 1])) * (knots[i + 1] - knots[i])
        self.assertEqual(repr(integral), repr(verify))


class TestModelicaMixinScaled(TestModelicaMixin):

    def setUp(self):
        self.problem = TestProblem()
        self.problem._nominals['x'] = 0.5
        self.problem._nominals['x_delayed'] = self.problem._nominals['x']
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6


class TestModelicaMixinNonConvex(TestCase):

    def setUp(self):
        self.tolerance = 1e-6

    def test_seeding(self):
        # Verify that both optima are found, depending on the seeding.
        self.problem = TestProblemNonConvex(np.ones(21) * 2.0)
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.assertAlmostEqual(self.results['x'][-1], 1.0, self.tolerance)

        self.problem = TestProblemNonConvex(np.ones(21) * -2.0)
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.assertAlmostEqual(self.results['x'][-1], -1.0, self.tolerance)


class TestModelicaMixinConstrained(TestCase):

    def setUp(self):
        self.problem = TestProblemConstrained()
        self.problem.optimize()
        self.results = self.problem.extract_results()

    def test_objective_value(self):
        # Make sure the constraint at t=1.9 has been applied.  With |u| <= 2, this ensures that
        # x(t=2.0)=0.0, the unconstrained optimum, can never be reached.
        x = self.problem.state_at('x', self.problem.times()[-1] - 0.1)

        self.assertAlmostGreaterThan(self.problem.objective_value, 1e-2, 0)
        self.assertAlmostEqual(self.results['u'][-1], -2, 1e-6)


class TestModelicaMixinTrapezoidal(TestCase):

    def setUp(self):
        self.problem = TestProblemTrapezoidal()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(
            abs(self.problem.objective_value), 0.0, objective_value_tol)


class TestModelicaMixinShort(TestCase):

    def setUp(self):
        self.problem = TestProblemShort()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(
            abs(self.problem.objective_value), 0.0, objective_value_tol)


class TestModelicaMixinAggregation(TestCase):

    def setUp(self):
        self.problem = TestProblemAggregation()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(
            abs(self.problem.objective_value), 0.0, objective_value_tol)

    def test_result_length(self):
        self.assertEqual(len(self.results['u']), 11)
        self.assertEqual(len(self.results['x']), 21)


class TestModelicaMixinEnsemble(TestCase):

    def setUp(self):
        self.problem = TestProblemEnsemble()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(
            abs(self.problem.objective_value), 0.0, objective_value_tol)


class TestModelicaMixinAlgebraic(TestCase):

    def setUp(self):
        self.problem = TestProblemAlgebraic()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_algebraic(self):
        self.assertAlmostEqual(self.results[
                               'y'] + self.results['u'], np.ones(len(self.problem.times())) * 1.0, self.tolerance)


class TestModelicaMixinMixedInteger(TestCase):

    def setUp(self):
        self.problem = TestProblemMixedInteger()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_booleans(self):
        self.assertAlmostEqual(self.results['choice'], np.zeros(
            21, dtype=np.bool), self.tolerance)
        self.assertAlmostEqual(self.results['other_choice'], np.ones(
            21, dtype=np.bool), self.tolerance)
        self.assertAlmostEqual(
            self.results['y'], -1 * np.ones(21, dtype=np.bool), self.tolerance)
