from rtctools.simulation.simulation_problem import SimulationProblem
import os
import re
import numpy as np
import collections

from .data_path import data_path
from test_case import TestCase

class SimulationTestProblem(SimulationProblem):
    def __init__(self):
        super().__init__(input_folder=data_path(), output_folder=data_path(
        ), model_name='TestModel', model_folder=data_path())

class TestSimulation(TestCase):
    def setUp(self):
        self.problem = SimulationTestProblem()

    def test_object(self):
        self.assertIsInstance(self.problem, SimulationTestProblem)
        
    def test_get_variables(self):
        all_variables = self.problem.get_variables()
        self.assertIsInstance(all_variables, collections.OrderedDict)

        self.assertEqual(set(all_variables),
            {'time', 'constant_input', 'k', 'switched', 'u', 'u_out',
             'w', 'der(w)', 'x', 'der(x)', 'x_start', 'y', 'z'})
        self.assertEqual(set(self.problem.get_parameter_variables()),
            {'x_start', 'k'})
        self.assertEqual(set(self.problem.get_input_variables()),
            {'constant_input', 'u'})
        self.assertEqual(set(self.problem.get_output_variables()),
            {'constant_output', 'switched', 'u_out', 'y', 'z'})

    def test_get_set_var(self):
        val = self.problem.get_var('switched')
        self.assertTrue(np.isnan(val))
        self.problem.set_var('switched', 10.0)
        val_reset = self.problem.get_var('switched')
        self.assertNotEqual(val_reset, val)

    def test_get_var_name_and_type(self):
        t = self.problem.get_var_type('switched')
        self.assertTrue(t == float)
        all_variables = self.problem.get_variables()
        idx = 0
        for var in all_variables.items():
            varname = var[0]
            if re.match(varname, "switched"):
                break
            idx += 1

        varname = self.problem.get_var_name(idx)
        self.assertEqual(varname, 'switched')

    def test_get_time(self):
        # test methods for get_time
        start = 0.0
        stop = 10.0
        dt = 1.0
        self.problem.setup_experiment(start, stop, dt)
        self.problem.set_var('x_start', 0.0)
        self.problem.set_var('constant_input', 0.0)
        self.problem.set_var('u', 0.0)
        self.problem.initialize()
        val = self.problem.get_start_time()
        self.assertAlmostEqual(self.problem.get_start_time(), start, 1e-6)
        self.assertAlmostEqual(self.problem.get_end_time(), stop, 1e-6)
        curtime = self.problem.get_current_time()
        self.assertAlmostEqual(curtime, start, 1e-6)
        while curtime < stop:
            self.problem.update(dt)
            curtime = self.problem.get_current_time()
        self.assertAlmostEqual(curtime, stop, 1e-6)

    def test_set_input(self):
        # run FMU model
        expected_values = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        stop = 1.0
        dt = 0.1
        self.problem.setup_experiment(0.0, stop, dt)
        self.problem.set_var('x_start', 0.25)
        self.problem.set_var('constant_input', 0.0)
        self.problem.set_var('u', 0.0)
        self.problem.initialize()
        i = 0
        while i < int(stop/dt):
            self.problem.set_var('u', 0.0)
            self.problem.update(dt)
            val = self.problem.get_var('switched')
            self.assertEqual(val, expected_values[i])
            i += 1

    def test_set_input2(self):
        # run FMU model
        expected_values = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        stop = 1.0
        dt = 0.1
        self.problem.setup_experiment(0.0, stop, dt)
        self.problem.set_var('x_start', 0.25)
        self.problem.set_var('constant_input', 0.0)
        self.problem.set_var('u', 0.0)
        self.problem.initialize()
        i = 0
        while i < int(stop/dt):
            self.problem.set_var('u', i)
            self.problem.update(dt)
            val = self.problem.get_var('u_out')
            self.assertEqual(val, i + 1)
            val = self.problem.get_var('switched')
            self.assertEqual(val, expected_values[i])
            i += 1
