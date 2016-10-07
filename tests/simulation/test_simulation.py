from rtctools.simulation.simulation_problem import SimulationProblem
from pyfmi.fmi_algorithm_drivers import FMICSAlgOptions
import os
import re
import numpy as np
import collections

from data_path import data_path
from test_case import TestCase

import pyfmi

class SimulationTestProblem(SimulationProblem):
    def __init__(self):
        # Call constructors
        SimulationProblem.__init__(self, data_path(), 'TestModel.fmu')

class TestSimulation(TestCase):
    def setUp(self):
        self.problem = SimulationTestProblem()

    def test_object(self):
        self.assertIsInstance(self.problem, SimulationTestProblem)

    def test_options(self):
        options = self.problem.get_options()
        self.assertIsInstance(options,FMICSAlgOptions)
        
    def test_get_variables(self):
        all_variables = self.problem.get_variables()
        self.assertIsInstance(all_variables, collections.OrderedDict)
        variables = []
        for var in all_variables.items():
            varname = var[0]
            # method returns all variables, including internal FMUvariables, starting with '_'
            if not re.match('_',varname):
                variables.append(varname)
        self.assertEqual(variables, ['alias', 'constant_input', 'constant_output', 'k', 'switched', 'u', 'w', 'der(w)', 'x', 'der(x)', 'x_delayed', 'y', 'z'] )
        nvar = self.problem.get_var_count()
        self.assertEqual(len(all_variables), nvar)

    def test_get_set_var(self):
        val = self.problem.get_var('switched')
        self.assertEqual(val, 0.0)
        self.problem.set_var('switched',10.0)
        val_reset = self.problem.get_var('switched')
        self.assertNotEqual(val_reset, val)

    def test_get_var_name_and_type(self):
        type = self.problem.get_var_type('switched')
        self.assertEqual(type, 'float')
        all_variables = self.problem.get_variables()
        idx = 0
        for var in all_variables.items():
            varname = var[0]
            if re.match(varname, "switched"):
                break
            idx += 1

        varname = self.problem.get_var_name(idx)
        self.assertEqual(varname,'switched')

    def test_get_time(self):
        # test methods for get_time
        start = 0.0
        stop = 10.0
        dt = 1.0
        self.problem.setup_experiment(start,stop, dt)
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

    def test_basic_run(self):
        # run FMU model 
        start = 0.0
        stop = 10.0
        dt = 1.0
        self.problem.setup_experiment(start,stop, dt)
        self.problem.initialize()
        curtime = self.problem.get_current_time()
        while curtime < stop:
            status = self.problem.update(dt)
            curtime = self.problem.get_current_time()
            print(curtime)
        self.problem.finalize()

    def test_set_input(self):
        # run FMU model
        self.problem.set_var('x',0.25)
        expected_values = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        stop = 1.0
        dt = 0.1
        self.problem.setup_experiment(0.0,stop,dt)
        self.problem.initialize()
        i = 0
        while i < int(stop/dt):
            status = self.problem.update(dt)
            val = self.problem.get_var('switched')
            self.assertEqual(val, expected_values[i])
            i += 1
        self.problem.finalize()
