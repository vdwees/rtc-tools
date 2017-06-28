from setuptools import setup
import rtctools
import sys

if sys.version_info[0] == 3 and sys.version_info[1] < 6:
    sys.exit("Sorry, Python 3.6 or newer is required.")

setup(
    name = 'rtctools',
    packages = ['rtctools', 'rtctools.data', 'rtctools.data.interpolation', 'rtctools.optimization', 'rtctools.simulation'],
    version = rtctools.__version__,
    description = 'Toolbox for control and optimization of environmental systems',
    author = 'Jorn Baayen',
    url = 'http://www.deltares.nl/en/software/rtc-tools/',
)