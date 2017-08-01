from rtctools.simulation.simulation_problem import SimulationProblem
from rtctools.simulation.pi_mixin import PIMixin
from pyfmi.fmi_algorithm_drivers import FMICSAlgOptions
import os
import re
import numpy as np
import collections

from data_path import data_path
from test_case import TestCase

import pyfmi

class SimulationTestProblem(PIMixin, SimulationProblem):
    #pi_validate_timeseries = False
    def __init__(self):
        super().__init__(input_folder=data_path(), output_folder=data_path(
        ), model_name='TestModel', model_folder=data_path())

class TestSimulation(TestCase):
    def setUp(self):
        self.problem = SimulationTestProblem()

    def test_simulate(self):
        self.problem.simulate()