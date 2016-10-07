# cython: embedsignature=True
"""
Commonly used utility functions.

Author:  Jorn Baayen
Date  :  March 16, 2016
"""

from casadi import CasadiOptions
import logging
import sys
import os
import re
import pstats
import cProfile

from data import pi
from optimization.pi_mixin import PIMixin
from . import __version__


def run_optimization_problem(optimization_problem_class, base_folder=None, log_level=logging.INFO, profile=False, profile_casadi=False):
    """
    Sets up and solves an optimization problem.

    This function makes the following assumptions:

    1. That the ``base_folder`` contains subfolders ``input``, ``output``, and ``model``, containing input data, output data, and the model, respectively.
    2. When using :class:`CSVLookupTableMixin`, that the base folder contains a subfolder ``lookup_tables``.
    3. When using :class:`ModelicaMixin`, that the base folder contains a subfolder ``model``.
    4. When using :class:`ModelicaMixin`, that the toplevel Modelica model name equals the class name.

    :param optimization_problem_class: Optimization problem class to solve.
    :param base_folder:                Base folder.
    :param log_level:                  The log level to use.
    :param profile:                    Whether or not to enable profiling.
    :param profile_casadi:             Whether or not to enable CasADi profiling.
    """

    if base_folder == None:
        # Check command line arguments
        if len(sys.argv) != 2:
            raise Exception("Usage: {} BASE_FOLDER".format(sys.argv[0]))

        base_folder = sys.argv[1]
    else:
        if not os.path.isabs(base_folder):
            # Resolve base folder relative to script folder
            base_folder = os.path.join(sys.path[0], base_folder)

    model_folder = os.path.join(base_folder, 'model')
    input_folder = os.path.join(base_folder, 'input')
    output_folder = os.path.join(base_folder, 'output')

    # Set up logging
    if issubclass(optimization_problem_class, PIMixin):
        handler = pi.DiagHandler(output_folder)
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
    logger = logging.getLogger("rtctools")
    logger.addHandler(handler)

    logger.setLevel(log_level)

    logger.info(
        "Using RTC-Tools {}, released as open source software under the GNU General Public License.".format(__version__))

    # Run
    try:
        prob = optimization_problem_class(
            model_folder=model_folder, input_folder=input_folder, output_folder=output_folder)
        if profile_casadi:
            # Use CasADi "profilereport" to process these results.
            filename = os.path.join(base_folder, 'profile_casadi.log')
            logger.info(
                "Logging CasADi profiling output to {}.  Use 'profilereport' to analyze the results.".format(filename))

            CasadiOptions.startProfiling(filename)
        if profile:
            # Must prepend set Cython compiler option "profile=True".
            logger.warning(
                "To profile effectively, compile RTC-Tools with the Cython compiler option 'profile=True'")

            filename = os.path.join(base_folder, "profile.prof")

            cProfile.runctx("prob.optimize()", globals(), locals(), filename)
            
            s = pstats.Stats(filename)
            s.strip_dirs().sort_stats("time").print_stats()
        else:
            prob.optimize()
    except TypeError:
        exc_info = sys.exc_info()
        value = exc_info[1]
        try:
            failed_class = re.search(
                "Can't instantiate (.*) with abstract methods", str(value)).group(1)
            abstract_method = re.search(
                ' with abstract methods (.*)', str(value)).group(1)
            logger.error("The {} is missing a mixin. Please add a mixin that instantiates abstract method {}, so that the optimizer can run.".format(
                failed_class, abstract_method))
        except:
            raise exc_info[0], exc_info[1], exc_info[2]


def run_simulation_problem(simulation_problem_class, base_folder=None, log_level=logging.INFO):
    """
    Sets up and runs a simulation problem.

    :param simulation_problem_class: Optimization problem class to solve.
    :param base_folder:              Folder within which subfolders "input", "output", and "model" exist, containing input and output data, and the model, respectively.
    :param log_level:                The log level to use.
    """

    if base_folder == None:
        # Check command line arguments
        if len(sys.argv) != 2:
            raise Exception("Usage: {} BASE_FOLDER".format(sys.argv[0]))

        base_folder = sys.argv[1]
    else:
        if not os.path.isabs(base_folder):
            # Resolve base folder relative to script folder
            base_folder = os.path.join(sys.path[0], base_folder)

    model_folder = os.path.join(base_folder, 'model')
    input_folder = os.path.join(base_folder, 'input')
    output_folder = os.path.join(base_folder, 'output')

    # Set up logging
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger("rtctools")
    logger.addHandler(handler)

    logger.setLevel(log_level)

    logger.info(
        "Using RTC-Tools {}, released as open source software under the GNU General Public License.".format(__version__))

    # Run
    prob = simulation_problem_class(
        model_folder=model_folder, input_folder=input_folder, output_folder=output_folder)
    prob.simulate()
