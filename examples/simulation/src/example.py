from rtctools.simulation.simulation_problem import SimulationProblem
from rtctools.simulation.csv_mixin import CSVMixin
from rtctools.util import run_simulation_problem

import numpy as np

import logging
logger = logging.getLogger("rtctools")

# A helper function for finding the closest value in an array
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

class Example(CSVMixin, SimulationProblem):
    """
    A basic example for introducing users to RTC-Tools 2 Simulation
    """

    # Discrete stages that the storage is capable of releasing
    release_stages = [0., 1., 2., 4., 8.] # m^3/s

    # Here is an example of overriding the update() method to show how control
    # can be build into the python script
    def update(self, dt):

        # Get the time step
        if dt < 0:
            dt = self._dt

        # Get relevant model variables
        volume = self.get_var('storage.V')
        target = self.get_var('storage_V_target')

        # Calucate error in storage.V
        error = target - volume

        # Calculate the desired control
        control = -error / dt

        # Get the closest feasible setting. 
        discrete_control = find_nearest(self.release_stages, control)

        # Set the control variable as the control for the next step of the simulation
        self.set_var('P_control', discrete_control)

        # Call the super class so that everything else continues as normal
    	super(Example, self).update(dt)

# Run
run_simulation_problem(Example, log_level=logging.DEBUG)