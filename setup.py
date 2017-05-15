from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import rtctools
import sys

if sys.version_info[0] == 2 and sys.version_info[1] < 7:
    sys.exit("Sorry, Python 2.7 or newer is required.")

setup(
    name = 'rtctools',
    packages = ['rtctools', 'rtctools.data', 'rtctools.data.interpolation', 'rtctools.optimization', 'rtctools.simulation'],
    version = rtctools.__version__,
    description = 'Toolbox for control and optimization of environmental systems',
    author = 'Jorn Baayen',
    url = 'http://www.deltares.nl/en/software/rtc-tools/',
)