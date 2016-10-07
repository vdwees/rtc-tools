from data_path import data_path
from test_case import TestCase

from rtctools.optimization.collocated_integrated_optimization_problem import CollocatedIntegratedOptimizationProblem
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.control_tree_mixin import ControlTreeMixin
from rtctools.optimization.timeseries import Timeseries
from casadi import MX, MXFunction
import numpy as np
import logging
import time
import sys
import os

logger = logging.getLogger("rtctools")
logger.setLevel(logging.DEBUG)


class TestProblem(ControlTreeMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):

    def __init__(self):
        super(TestProblem, self).__init__(
            model_name='TestModelWithInitial', model_folder=data_path())

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    def pre(self):
        # Do nothing
        pass

    def delayed_feedback(self):
        # Delayed feedback
        return [('x', 'x_delayed', 0.1)]

    @property
    def ensemble_size(self):
        return 2

    def control_tree_options(self):
        return {'forecast_variables': ['constant_input'],
                'branching_times': [0.1, 0.2],
                'k': 3}

    def constant_inputs(self, ensemble_member):
        # Constant inputs
        if ensemble_member == 0:
            return {'constant_input': Timeseries(self.times(), np.linspace(1.0, 0.0, 21))}
        else:
            return {'constant_input': Timeseries(self.times(), np.linspace(1.0, 0.5, 21))}

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


class TestControlTreeMixin(TestCase):

    def setUp(self):
        self.problem = TestProblem()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(
            abs(self.problem.objective_value), 0.0, objective_value_tol)
