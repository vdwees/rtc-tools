# cython: embedsignature=True

from rtctools.data.interpolation.bspline1d import BSpline1D
from rtctools.data.interpolation.bspline2d import BSpline2D
from optimization_problem import OptimizationProblem, LookupTable
from scipy.interpolate import splrep, bisplrep, splev, bisplev
from casadi import SX, SXFunction
import numpy as np
cimport numpy as np
import ConfigParser
import logging
import cython
import glob
import os
import sys
import rtctools.data.csv as csv

import cPickle as pickle

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

    #: Cashe Lookup Tables
    use_cached_lookup_tables = False


    def __init__(self, **kwargs):
        # Check arguments
        if 'input_folder' in kwargs:
            assert('lookup_table_folder' not in kwargs)

            self._lookup_table_folder = os.path.join(
                kwargs['input_folder'], 'lookup_tables')
        else:
            self._lookup_table_folder = kwargs['lookup_table_folder']

        # Call parent
        super(CSVLookupTableMixin, self).__init__(**kwargs)

    def pre(self):
        # Call parent class first for default behaviour.
        super(CSVLookupTableMixin, self).pre()

        # Get curve fitting options from curvefit_options.ini file
        ini_path = os.path.join(
            self._lookup_table_folder, 'curvefit_options.ini')
        try:
            ini_config = ConfigParser.RawConfigParser()
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
                except ConfigParser.NoSectionError:
                    prop = 0
                except ConfigParser.NoOptionError:
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
        self._lookup_tables = {}
        for filename in glob.glob(os.path.join(self._lookup_table_folder, "*.csv")):

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
            # If use_cached_lookup_tables is True, first try to load the cashed tck
            if self.use_cached_lookup_tables is True:
                logger.debug('CSVLookupTableMixin: Attempting to use cashed tck values for {}'.format(output))
                try:
                    tck = pickle.load(open(filename.replace('.csv', '.tck'), 'rb'))
                except IOError:
                    logger.info('CSVLookupTableMixin: Cashed tck values for {} not found'.format(output))
                    tck = None

            if len(csvinput.dtype.names) == 2:
                if tck is None:
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
                symbols = [SX.sym(inputs[0])]
                function = SXFunction(symbols, [BSpline1D(*tck)(symbols[0])])
                function.init()
                self._lookup_tables[output] = LookupTable(symbols, function)

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
                symbols = [SX.sym(inputs[0]), SX.sym(inputs[1])]
                function = SXFunction(
                    symbols, [BSpline2D(*tck)(symbols[0], symbols[1])])
                function.init()
                self._lookup_tables[output] = LookupTable(symbols, function)

            else:
                raise Exception(
                    "CSVLookupTableMixin: {}-dimensional lookup tables not implemented yet.".format(len(csvinput.dtype.names)))

            if self.use_cached_lookup_tables is True:
                pickle.dump(tck, open(filename.replace('.csv', '.tck'), 'wb'))

    def lookup_tables(self, ensemble_member):
        # Call parent class first for default values.
        lookup_tables = super(CSVLookupTableMixin,
                              self).lookup_tables(ensemble_member)

        # Update lookup_tables with imported csv lookup tables
        lookup_tables.update(self._lookup_tables)

        return lookup_tables
