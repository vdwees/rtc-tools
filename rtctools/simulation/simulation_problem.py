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

    # Folder in which the referenced Modelica libraries are found
    modelica_library_folder = os.getenv('DELTARES_LIBRARY_PATH', 'mo')

    def __init__(self, **kwargs):
        # Check arguments
        assert('model_folder' in kwargs)

        # Determine the name of the model
        if 'model_name' in kwargs:
            model_name = kwargs['model_name']
        else:
            if hasattr(self, 'model_name'):
                model_name = self.model_name
            else:
                model_name = self.__class__.__name__

        # Load the FMU, compiling it if needed
        model_folder = kwargs['model_folder']
        if not os.path.isdir(model_folder):
            raise RuntimeError("Directory does not exist" + model_folder)

        need_compilation = False

        fmu_filename = os.path.join(model_folder, model_name + '.fmu')
        if os.path.isfile(fmu_filename):
            fmu_mtime = os.path.getmtime(fmu_filename)
        else:
            need_compilation = True

        mo_filenames = []
        for f in os.listdir(model_folder):
            if f.endswith(".mo"):
                mo_filename = os.path.join(model_folder, f)
                mo_filenames.append(mo_filename)

                if not need_compilation and os.path.getmtime(mo_filename) > fmu_mtime:
                    need_compilation = True

        if need_compilation:
            # compile .mo files into .fmu
            logger.info("Compiling FMU")

            try:
                compile_fmu(model_name, mo_filenames, version=2.0, target='cs',
                            compiler_options=self.compiler_options(), compiler_log_level='i:compile_fmu_log.txt',
                            compile_to=fmu_filename)
            except ModelicaClassNotFoundError:
                raise RuntimeError("Could not find files to compile FMU.")

        self.__model = pyfmi.load_fmu(fmu_filename)
        if self.__model is None:
            raise RuntimeError("FMU could not be loaded")
        self.__model_types = {0: 'float', 1: 'int',
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
        self.__model.initialize()

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

    def setup_experiment(self, start, stop, dt=-1, tol=None):
        """ 
        Create an experiment.

        :param start: Start time for the simulation.
        :param stop:  Final time for the simulation.
        :param dt:    Time step size.
        :param tol:   Tolerance of the underlying FMU method.
        """
        if tol is None:
            tol = self.__model.get_default_experiment_tolerance()
        self.__start = start
        self.__stop = stop
        self.__dt = dt
        self.__model.setup_experiment(tol, tol, start, stop, stop)

    def finalize(self):
        """
        Finalize FMU.
        """
        self.__model.terminate()

    def update(self, dt):
        """
        Performs one timestep. 

        The method ``setup_experiment`` must have been called before.

        :param dt: Time step size.
        """
        if dt < 0:
            dt = self.__dt

        logger.debug("Taking a step at {} with size {}".format(self.get_current_time(), dt))
        return self.__model.do_step(self.__model.time, dt, True)

    def simulate(self):
        """ 
        Run model from start_time to end_time.
        """

        # Do any preprocessing, which may include changing parameter values on
        # the model
        logger.info("Preprocessing")
        self.pre()

        # Initialize model
        logger.info("Initializing FMU")
        self.initialize()

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
        self.__model.reset()

    def get_start_time(self):
        """
        Return start time of experiment.

        :returns: The start time of the experiment.
        """
        return self.__start

    def get_end_time(self):
        """
        Return end time of experiment.

        :returns: The end time of the experiment.
        """
        return self.__stop

    def get_current_time(self):
        """
        Return current time of simulation.

        :returns: The current simulation time.
        """
        return self.__model.time

    def get_options(self):
        """
        Return the available options of the FMU.

        :returns: A dictionary of options supported by the FMU.
        """
        return self.__model.simulate_options()

    def get_var(self, name):
        """
        Return a numpy array from FMU.

        :param name: Variable name.

        :returns: The value of the variable.
        """
        return self.__model.get(name)

    def get_var_count(self):
        """
        Return the number of variables (internal FMU and user declared).

        :returns: The number of variables supported by the FMU.
        """
        return len(self.__model.get_model_variables())

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
        retval = self.__model.get_variable_data_type(name)
        return self.__model_types[retval]

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
        return self.__model.get_model_variables()

    def get_parameter_variables(self):
        return self.__model.get_model_variables(causality=1)

    def get_input_variables(self):
        return self.__model.get_model_variables(causality=2)

    def get_output_variables(self):
        return self.__model.get_model_variables(causality=3)

    def set_var(self, name, val):
        """
        Set the value of the given variable.

        :param name: Name of variable to set.
        :param val:  Value(s).
        """
        self.__model.set(name, val)

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

    def compiler_options(self):
        """
        Subclasses can configure the `JModelica.org <http://www.jmodelica.org/>`_ compiler options here.

        :returns: A dictionary of JModelica.org compiler options.  See the JModelica.org documentation for details.
        """

        # Default options
        compiler_options = {}

        # No automatic division with variables please.  Our variables may
        # sometimes equal to zero.
        compiler_options['divide_by_vars_in_tearing'] = False

        # Include the 'mo' folder as library dir by default.
        compiler_options['extra_lib_dirs'] = self.modelica_library_folder

        # Done
        return compiler_options
