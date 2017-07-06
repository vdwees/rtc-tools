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
    release_stages = [0., 1., 2., 4., 8.]

    # Here is an example of overriding the update() method to show how control
    # can be build into the python script
    def update(self, dt):

    	# We implemented a PI controller in the model. We like that it keeps
    	# track of integrals and error, but our release has discrete stages
    	# and bounds that the simple controller does not capture. So we
    	# process the control output in the python script.

        # Get the time step
        if dt < 0:
            dt = self._dt

    	# Get the controller output. It is negative in the model, and we
    	# divide by dt to get in terms of instantaneous flow rates (m^3/s)
        control_var = -self.get_var('PI.control') / dt

        # Get the closest feasible setting. 
        discrete_control_var = find_nearest(self.release_stages, control_var)

        # Set the control variable as the control for the next step of the simulation
        self.set_var('PI_control', discrete_control_var)

        # Call the super class so that everything else continues as normal
    	super(Example, self).update(dt)

# Run
run_simulation_problem(Example, log_level=logging.DEBUG)