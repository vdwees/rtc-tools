from rtctools.simulation.simulation_problem import SimulationProblem
from rtctools.simulation.csv_mixin import CSVMixin
from rtctools.util import run_simulation_problem

import logging
logger = logging.getLogger("rtctools")

class Example(CSVMixin, SimulationProblem):
    """
    A basic example for introducing users to RTC-Tools 2 Simulation
    """

# Run
run_simulation_problem(Example, log_level=logging.DEBUG)