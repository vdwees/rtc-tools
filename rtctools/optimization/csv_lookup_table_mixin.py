from rtctools.data.interpolation.bspline1d import BSpline1D
from rtctools.data.interpolation.bspline2d import BSpline2D
import rtctools.data.csv as csv
from scipy.interpolate import splrep, bisplrep, splev, bisplev
import casadi as ca
import numpy as np
import configparser
import logging
import pickle
import glob
import os
import sys

from .optimization_problem import OptimizationProblem, LookupTable

logger = logging.getLogger("rtctools")


class CSVLookupTableMixin(OptimizationProblem):
    """
    Adds lookup tables to your optimization problem.

    During preprocessing, the CSV files located inside the ``lookup_tables`` subfolder are read.
    In every CSV file, the first column contains the output of the lookup table.  Subsequent columns contain
    the input variables.

    Cubic B-Splines are used to turn the data points into continuous lookup tables.

    Optionally, a file ``curvefit_options.ini`` may be included inside the ``lookup_tables`` folder. This file contains,
    grouped per lookup table, the following options:

    * monotonicity:
        * is an integer, magnitude is ignored
        * if positive, causes spline to be monotonically increasing
        * if negative, causes spline to be monotonically decreasing
        * if 0, leaves spline monotonicity unconstrained

    * curvature:
        * is an integer, magnitude is ignored
        * if positive, causes spline curvature to be positive (convex)
        * if negative, causes spline curvature to be negative (concave)
        * if 0, leaves spline curvature unconstrained

    .. note:: Currently only one-dimensional lookup tables are fully supported.  Support for two-dimensional lookup tables is experimental.

    :cvar csv_delimiter:                 Column delimiter used in CSV files.  Default is ``,``.
    :cvar csv_lookup_table_debug:        Whether to generate plots of the spline fits.  Default is ``false``.
    :cvar csv_lookup_table_debug_points: Number of evaluation points for plots.  Default is ``100``.
    """

    #: Column delimiter used in CSV files
    csv_delimiter = ','

    #: Debug settings
    csv_lookup_table_debug = False
    csv_lookup_table_debug_points = 100

    def __init__(self, **kwargs):
        # Check arguments
        if 'input_folder' in kwargs:
            assert('lookup_table_folder' not in kwargs)

            self.__lookup_table_folder = os.path.join(
                kwargs['input_folder'], 'lookup_tables')
        else:
            self.__lookup_table_folder = kwargs['lookup_table_folder']

        # Call parent
        super().__init__(**kwargs)

    def pre(self):
        # Call parent class first for default behaviour.
        super().pre()

        # Get curve fitting options from curvefit_options.ini file
        ini_path = os.path.join(
            self.__lookup_table_folder, 'curvefit_options.ini')
        try:
            ini_config = configparser.RawConfigParser()
            ini_config.readfp(open(ini_path))
            no_curvefit_options = False
        except IOError:
            logger.info(
                "CSVLookupTableMixin: No curvefit_options.ini file found. Using default values.")
            no_curvefit_options = True

        def get_curvefit_options(curve_name, no_curvefit_options=no_curvefit_options):
            if no_curvefit_options:
                return 0, 0, 0

            curvefit_options = []

            def get_property(prop_name):
                try:
                    prop = int(ini_config.get(curve_name, prop_name))
                except configparser.NoSectionError:
                    prop = 0
                except configparser.NoOptionError:
                    prop = 0
                except ValueError:
                    raise Exception(
                        "CSVLookupTableMixin: Invalid {0} constraint for {1}. {0} should be either -1, 0, or 1.".format(prop_name, curve_name))
                return prop

            for prop_name in ['monotonicity', 'monotonicity2', 'curvature']:
                curvefit_options.append(get_property(prop_name))

            logger.debug("CSVLookupTableMixin: Curve fit option for {}:({},{},{})".format(
                curve_name, *curvefit_options))
            return tuple(curvefit_options)

        # Read CSV files
        logger.info(
            "CSVLookupTableMixin: Generating Splines from lookup table data.")
        self.__lookup_tables = {}
        for filename in glob.glob(os.path.join(self.__lookup_table_folder, "*.csv")):

            logger.debug(
                "CSVLookupTableMixin: Reading lookup table from {}".format(filename))

            csvinput = csv.load(filename, delimiter=self.csv_delimiter)
            output = csvinput.dtype.names[0]
            inputs = csvinput.dtype.names[1:]

            # Get monotonicity and curvature from ini file
            mono, mono2, curv = get_curvefit_options(output)

            logger.debug(
                "CSVLookupTableMixin: Output is {}, inputs are {}.".format(output, inputs))

            tck = None
            # If tck file is newer than the csv file, first try to load the cached values from the tck file
            tck_filename = filename.replace('.csv', '.tck')
            valid_cache = False
            if os.path.exists(tck_filename):
                if no_curvefit_options:
                    valid_cache = os.path.getmtime(filename) < os.path.getmtime(tck_filename)
                else:
                    valid_cache = (os.path.getmtime(filename) < os.path.getmtime(tck_filename)) and \
                                  (os.path.getmtime(ini_path) < os.path.getmtime(tck_filename))
                if valid_cache:
                    logger.debug(
                        'CSVLookupTableMixin: Attempting to use cached tck values for {}'.format(output))
                    with open(tck_filename, 'rb') as f:
                        try:
                            tck = pickle.load(f)
                        except OSError:
                            valid_cache = False
            if not valid_cache:
                logger.info(
                    'CSVLookupTableMixin: Recalculating tck values for {}'.format(output))

            if len(csvinput.dtype.names) == 2:
                if not valid_cache:
                    k = 3  # default value
                    # 1D spline fitting needs k+1 data points
                    if len(csvinput[output]) >= k + 1:
                        tck = BSpline1D.fit(csvinput[inputs[0]], csvinput[
                                            output], k=k, monotonicity=mono, curvature=curv)
                    else:
                        raise Exception(
                            "CSVLookupTableMixin: Too few data points in {} to do spline fitting. Need at least {} points.".format(filename, k + 1))

                if self.csv_lookup_table_debug:
                    import pylab
                    i = np.linspace(csvinput[inputs[0]][0], csvinput[
                                    inputs[0]][-1], self.csv_lookup_table_debug_points)
                    o = splev(i, tck)
                    pylab.clf()
                    # TODO: Figure out why cross-section B0607 in NZV does not
                    # conform to constraints!
                    pylab.plot(i, o)
                    pylab.plot(csvinput[inputs[0]], csvinput[
                               output], linestyle='', marker='x', markersize=10)
                    figure_filename = filename.replace('.csv', '.png')
                    pylab.savefig(figure_filename)
                symbols = [ca.SX.sym(inputs[0])]
                function = ca.Function('f', symbols, [BSpline1D(*tck)(symbols[0])])
                self.__lookup_tables[output] = LookupTable(symbols, function)

            elif len(csvinput.dtype.names) == 3:
                if tck is None:
                    kx = 3  # default value
                    ky = 3  # default value

                    # 2D spline fitting needs (kx+1)*(ky+1) data points
                    if len(csvinput[output]) >= (kx + 1) * (ky + 1):
                        # TODO: add curvature paramenters from curvefit_options.ini
                        # once 2d fit is implemented
                        tck = bisplrep(csvinput[inputs[0]], csvinput[
                                       inputs[1]], csvinput[output], kx=kx, ky=ky)
                    else:
                        raise Exception("CSVLookupTableMixin: Too few data points in {} to do spline fitting. Need at least {} points.".format(
                            filename, (kx + 1) * (ky + 1)))

                if self.csv_lookup_table_debug:
                    import pylab
                    i1 = np.linspace(csvinput[inputs[0]][0], csvinput[
                                     inputs[0]][-1], self.csv_lookup_table_debug_points)
                    i2 = np.linspace(csvinput[inputs[1]][0], csvinput[
                                     inputs[1]][-1], self.csv_lookup_table_debug_points)
                    i1, i2 = np.meshgrid(i1, i2)
                    i1 = i1.flatten()
                    i2 = i2.flatten()
                    o = bisplev(i1, i2, tck)
                    pylab.clf()
                    pylab.plot_surface(i1, i2, o)
                    figure_filename = filename.replace('.csv', '.png')
                    pylab.savefig(figure_filename)
                symbols = [ca.SX.sym(inputs[0]), ca.SX.sym(inputs[1])]
                function = ca.Function('f', 
                    symbols, [BSpline2D(*tck)(symbols[0], symbols[1])])
                self.__lookup_tables[output] = LookupTable(symbols, function)

            else:
                raise Exception(
                    "CSVLookupTableMixin: {}-dimensional lookup tables not implemented yet.".format(len(csvinput.dtype.names)))

            if not valid_cache:
                pickle.dump(tck, open(filename.replace('.csv', '.tck'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def lookup_tables(self, ensemble_member):
        # Call parent class first for default values.
        lookup_tables = super(CSVLookupTableMixin,
                              self).lookup_tables(ensemble_member)

        # Update lookup_tables with imported csv lookup tables
        lookup_tables.update(self.__lookup_tables)

        return lookup_tables
