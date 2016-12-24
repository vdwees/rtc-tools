# cython: embedsignature=True

import os
import logging
import numpy as np
import pyfmi.fmi
from pymodelica.compiler import compile_fmu
from pymodelica.compiler_exceptions import *

logger = logging.getLogger("rtctools")


class SimulationProblem:
    """
    `FMU <https://fmi-standard.org/>`_ simulation runner.
    
    Implements the `BMI <http://csdms.colorado.edu/wiki/BMI_Description>`_ Interface.
    """

    def __init__(self, model_folder, model_name):
        """
        Constructor.

        :param model_folder:    path to directory containing either the FMU file
                                or the source files to generate it.
        :param model_name:      FMU filename, including extension (.fmu); if it 
                                does not exist model_folder is searched for .mo files 
                                to compile the FMU on the fly.
        """
        if not os.path.isdir(model_folder):
            raise RuntimeError("Directory does not exist" + model_folder)
        if not os.path.isfile(os.path.join(model_folder, model_name)):
            # compile .mo files into .fmu
            os.chdir(model_folder)
            cwd = os.getcwd()
            mo_files = []
            for file in os.listdir(model_folder):
                if file.endswith(".mo"):
                    mo_files.append(file)
            try:
                compile_fmu(model_name.replace(".fmu", ""), mo_files, version=2.0, target='cs',
                            compiler_log_level='i:compile_fmu_log.txt')
                os.chdir(cwd)
            except ModelicaClassNotFoundError:
                raise RuntimeError("Could not find files to compile FMU.")

        self._model_folder = model_folder
        self._model_name = model_name
        self._model = pyfmi.load_fmu(os.path.join(model_folder, model_name))
        if self._model is None:
            raise RuntimeError("FMU could not be loaded")
        self._model_types = {0: 'float', 1: 'int',
                             2: 'bool', 3: 'str', 4: 'dict'}

    def initialize(self, config_file=None):
        """
        Initialize FMU with default values

        :param config_file: Path to an initialization file.
        """
        if config_file:
            # TODO read start and stop time from configfile and call:
            # self.setup_experiment(start,stop)
            # for now, assume that setup_experiment was called beforehand
            raise NotImplementedError
        self._model.initialize()

    def pre(self):
        """
        Any preprocessing takes place here.
        """
        pass

    def post(self):
        """
        Any postprocessing takes place here.
        """
        pass

    def setup_experiment(self, start, stop, dt, tol=None):
        """ 
        Create an experiment.

        :param start: Start time for the simulation.
        :param stop:  Final time for the simulation.
        :param dt:    Time step size.
        :param tol:   Tolerance of the underlying FMU method.
        """
        if tol is None:
            tol = self._model.get_default_experiment_tolerance()
        self._start = start
        self._stop = stop
        self._dt = dt
        self._model.setup_experiment(tol, tol, start, stop, stop)

    def finalize(self):
        """
        Finalize FMU.
        """
        self._model.terminate()

    def update(self, dt):
        """
        Performs one timestep. 

        The method ``setup_experiment`` must have been called before.

        :param dt: Time step size.
        """
        if dt < 0:
            dt = self._dt
        return self._model.do_step(self._model.time, dt, True)

    def simulate(self):
        """ 
        Run model from start_time to end_time.
        """

        # Do any preprocessing, which may include changing parameter values on
        # the model
        logger.info("Preprocessing")
        self.pre()

        # Perform all timesteps
        logger.info("Running FMU")
        while self.get_current_time() < self.get_end_time():
            self.update(-1)

        # Do any postprocessing
        logger.info("Postprocessing")
        self.post()

    def reset(self):
        """
        Reset the FMU.
        """
        self._model.reset()

    def get_start_time(self):
        """
        Return start time of experiment.

        :returns: The start time of the experiment.
        """
        return self._start

    def get_end_time(self):
        """
        Return end time of experiment.

        :returns: The end time of the experiment.
        """
        return self._stop

    def get_current_time(self):
        """
        Return current time of simulation.

        :returns: The current simulation time.
        """
        return self._model.time

    def get_options(self):
        """
        Return the available options of the FMU.

        :returns: A dictionary of options supported by the FMU.
        """
        return self._model.simulate_options()

    def get_var(self, name):
        """
        Return a numpy array from FMU.

        :param name: Variable name.

        :returns: The value of the variable.
        """
        return self._model.get(name)

    def get_var_count(self):
        """
        Return the number of variables (internal FMU and user declared).

        :returns: The number of variables supported by the FMU.
        """
        return len(self._model.get_model_variables())

    def get_var_name(self, i):
        """
        Returns the name of a variable.

        :param i: Index in ordered dictionary returned by FMU-method get_model_variables.

        :returns: The name of the variable.
        """
        return self.get_variables().items()[i][0]

    def get_var_type(self, name):
        """
        Return type string, compatible with numpy.

        :param name: Variable name.

        :returns: The type of the variable.
        """
        retval = self._model.get_variable_data_type(name)
        return self._model_types[retval]

    def get_var_rank(self, name):
        """
        Not implemented
        """
        raise NotImplementedError

    def get_var_shape(self, name):
        """
        Not implemented
        """
        raise NotImplementedError

    def get_variables(self):
        """
        Return all variables of FMU (both internal and user defined)

        :returns: A list of all variables supported by the FMU.
        """
        return self._model.get_model_variables()

    def set_var(self, name, val):
        """
        Set the value of the given variable.

        :param name: Name of variable to set.
        :param val:  Value(s).
        """
        self._model.set(name, val)

    def set_var_slice(self, name, start, count, var):
        """
        Not implemented.
        """
        raise NotImplementedError

    def set_var_index(self, name, index, var):
        """
        Not implemented.
        """
        raise NotImplementedError

    def inq_compound(self, name):
        """
        Not implemented.
        """
        raise NotImplementedError

    def inq_compound_field(self, name, index):
        """
        Not implemented.
        """
        raise NotImplementedError
