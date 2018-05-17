import cProfile
import logging
import os
import pstats
import re
import sys

import casadi

from . import __version__
from ._internal.alias_tools import OrderedSet
from .data import pi
from .optimization.pi_mixin import PIMixin


def run_optimization_problem(optimization_problem_class, base_folder='..', log_level=logging.INFO, profile=False):
    """
    Sets up and solves an optimization problem.

    This function makes the following assumptions:

    1. That the ``base_folder`` contains subfolders ``input``, ``output``, and ``model``,
       containing input data, output data, and the model, respectively.
    2. When using :class:`CSVLookupTableMixin`, that the base folder contains a subfolder ``lookup_tables``.
    3. When using :class:`ModelicaMixin`, that the base folder contains a subfolder ``model``.
    4. When using :class:`ModelicaMixin`, that the toplevel Modelica model name equals the class name.

    :param optimization_problem_class: Optimization problem class to solve.
    :param base_folder:                Base folder.
    :param log_level:                  The log level to use.
    :param profile:                    Whether or not to enable profiling.

    :returns: :class:`OptimizationProblem` instance.
    """

    if not os.path.isabs(base_folder):
        # Resolve base folder relative to script folder
        base_folder = os.path.join(sys.path[0], base_folder)

    model_folder = os.path.join(base_folder, 'model')
    input_folder = os.path.join(base_folder, 'input')
    output_folder = os.path.join(base_folder, 'output')

    # Set up logging
    logger = logging.getLogger("rtctools")

    # Add stream handler if it does not already exist.
    if not logger.hasHandlers() and not any((isinstance(h, logging.StreamHandler) for h in logger.handlers)):
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Add pi.DiagHandler, if using PIMixin. Only add it if it does not already exist.
    if (issubclass(optimization_problem_class, PIMixin) and
            not any((isinstance(h, pi.DiagHandler) for h in logger.handlers))):
        handler = pi.DiagHandler(output_folder)
        logger.addHandler(handler)

    # Set log level
    logger.setLevel(log_level)

    # Log version info
    logger.info(
        "Using RTC-Tools {}.".format(__version__))
    logger.debug(
        "Using CasADi {}.".format(casadi.__version__))

    # Check for some common mistakes in inheritance order
    suggested_order = OrderedSet([
        'HomotopyMixin', 'GoalProgrammingMixin', 'PIMixin', 'CSVMixin',
        'ModelicaMixin', 'CollocatedIntegratedOptimizationProblem', 'OptimizationProblem'])
    base_names = OrderedSet([b.__name__ for b in optimization_problem_class.__bases__])
    if suggested_order & base_names != base_names & suggested_order:
        msg = 'Please inherit from base classes in the following order: {}'.format(list(base_names & suggested_order))
        logger.error(msg)
        raise Exception(msg)

    # Run
    try:
        prob = optimization_problem_class(
            model_folder=model_folder, input_folder=input_folder, output_folder=output_folder)
        if profile:
            filename = os.path.join(base_folder, "profile.prof")

            cProfile.runctx("prob.optimize()", globals(), locals(), filename)

            s = pstats.Stats(filename)
            s.strip_dirs().sort_stats("time").print_stats()
        else:
            prob.optimize()
        return prob
    except Exception as e:
        logger.error(str(e))
        if isinstance(e, TypeError):
            exc_info = sys.exc_info()
            value = exc_info[1]
            try:
                failed_class = re.search(
                    "Can't instantiate (.*) with abstract methods", str(value)).group(1)
                abstract_method = re.search(
                    ' with abstract methods (.*)', str(value)).group(1)
                logger.error(
                    'The {} is missing a mixin. Please add a mixin that instantiates '
                    'abstract method {}, so that the optimizer can run.'.format(
                        failed_class, abstract_method))
            except Exception:
                pass
        for handler in logger.handlers:
            handler.flush()
        raise


def run_simulation_problem(simulation_problem_class, base_folder='..', log_level=logging.INFO):
    """
    Sets up and runs a simulation problem.

    :param simulation_problem_class: Optimization problem class to solve.
    :param base_folder:              Folder within which subfolders "input", "output", and "model" exist,
                                     containing input and output data, and the model, respectively.
    :param log_level:                The log level to use.

    :returns: :class:`SimulationProblem` instance.
    """

    if base_folder is None:
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
    logger = logging.getLogger("rtctools")
    if not any((isinstance(h, logging.StreamHandler) for h in logger.handlers)):
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(log_level)

    logger.info(
        'Using RTC-Tools {}, released as open source software'
        'under the GNU General Public License.'.format(__version__))
    logger.debug(
        'Using CasADi {}.'.format(casadi.__version__))

    # Run
    prob = simulation_problem_class(
        model_folder=model_folder, input_folder=input_folder, output_folder=output_folder)
    prob.simulate()
    return prob
