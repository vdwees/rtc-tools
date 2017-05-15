from data_path import data_path
from test_case import TestCase

from rtctools.optimization.collocated_integrated_optimization_problem import CollocatedIntegratedOptimizationProblem
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries
from casadi import MX
import numpy as np
import logging
import time
import sys
import os

logger = logging.getLogger("rtctools")


class HybridShootingTestProblem(ModelicaMixin, CollocatedIntegratedOptimizationProblem):

    def __init__(self, integrated_states):
        super(HybridShootingTestProblem, self).__init__(input_folder=data_path(
        ), output_folder=data_path(), model_name='HybridShootingTestModel', model_folder=data_path())

        self._integrated_states = integrated_states

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    @property
    def integrated_states(self):
        return self._integrated_states

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


class TestHybridShooting(TestCase):

    def setUp(self):
        self.problem = HybridShootingTestProblem([])
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(
            abs(self.problem.objective_value), 0.0, objective_value_tol)


class TestHybridShootingX(TestHybridShooting):

    def setUp(self):
        self.problem = HybridShootingTestProblem(['x'])
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6


class TestHybridShootingW(TestHybridShooting):

    def setUp(self):
        self.problem = HybridShootingTestProblem(['w'])
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6


class TestSingleShooting(TestHybridShooting):

    def setUp(self):
        self.problem = HybridShootingTestProblem(['x', 'w'])
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6
