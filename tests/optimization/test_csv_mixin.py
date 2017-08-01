from .data_path import data_path
from test_case import TestCase

from rtctools.optimization.collocated_integrated_optimization_problem import CollocatedIntegratedOptimizationProblem
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.csv_lookup_table_mixin import CSVLookupTableMixin
from casadi import MX
import numpy as np
import logging
import time
import sys
import os


logger = logging.getLogger("rtctools")
logger.setLevel(logging.WARNING)


class TestProblem(CSVMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):

    def __init__(self, **kwargs):
        kwargs['model_name'] = kwargs.get('model_name', 'TestModel')
        kwargs['input_folder'] = data_path()
        kwargs['output_folder'] = data_path()
        kwargs['model_folder'] = data_path()
        super().__init__(**kwargs)

    def delayed_feedback(self):
        # Delayed feedback
        return [('x', 'x_delayed', 0.1)]

    def objective(self, ensemble_member):
        # Quadratic penalty on state 'x' at final time
        xf = self.state_at('x', self.times()[-1])
        f = xf**2
        return f

    def constraints(self, ensemble_member):
        # No additional constraints
        return []

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options['cache'] = False
        return compiler_options


class TestProblemLookup(CSVLookupTableMixin, TestProblem):

    def __init__(self):
        super().__init__(input_folder=data_path(), output_folder=data_path(
        ), model_name='TestModel', model_folder=data_path(), lookup_tables=['constant_input'])


class TestProblemEnsemble(TestProblem):

    csv_ensemble_mode = True

    def __init__(self):
        super().__init__(input_folder=data_path(), output_folder=data_path(
        ), model_name='TestModel', model_folder=data_path(), lookup_tables=[])


class TestCSVMixin(TestCase):

    def setUp(self):
        self.problem = TestProblem()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_parameter(self):
        params = self.problem.parameters(0)
        self.assertEqual(params['k'], 1.01)

    def test_initial_state(self):
        initial_state = self.problem.initial_state(0)
        self.assertAlmostEqual(initial_state['x'], 1.02, self.tolerance)

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertTrue(abs(self.problem.objective_value)
                        < objective_value_tol)

    def test_output(self):
        self.assertAlmostEqual(self.results['x'][
                               :]**2 + np.sin(self.problem.times()), self.results['z'][:], self.tolerance)

    def test_algebraic(self):
        self.assertAlmostEqual(self.results[
                               'y'] + self.results['x'], np.ones(len(self.problem.times())) * 3.0, self.tolerance)

    def test_bounds(self):
        self.assertAlmostGreaterThan(self.results['u'], -2, self.tolerance)
        self.assertAlmostLessThan(self.results['u'], 2, self.tolerance)

    def test_interpolate(self):
        for v in ['x', 'y', 'u']:
            for i in [0, int(len(self.problem.times()) / 2), -1]:
                a = self.problem.interpolate(
                    self.problem.times()[i], self.problem.times(), self.results[v], 0.0, 0.0)
                b = self.results[v][i]
                self.assertAlmostEqual(a, b, self.tolerance)


class TestCSVLookupMixin(TestCSVMixin):

    def setUp(self):
        self.problem = TestProblemLookup()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_call(self):
        self.assertAlmostEqual(self.problem.lookup_tables(
            0)['constant_input'](0.2), 2.0, self.tolerance)
        self.assertAlmostEqual(self.problem.lookup_tables(0)['constant_input'](
            np.array([0.2, 0.3])), np.array([2.0, 3.0]), self.tolerance)


class TestPIMixinEnsemble(TestCase):

    def setUp(self):
        self.problem = TestProblemEnsemble()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertTrue(abs(self.problem.objective_value)
                        < objective_value_tol)
